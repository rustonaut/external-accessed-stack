//! This ways provides a way to allow access to a (async) stack allocated value outside
//! of the given async task and/or thread.
//!
//! The main use of this library is to provide a way to have a stack allocated*
//! buffer which can be used by completion base I/O like DMA.
//!
//! In many cases it's much easier with a much better API UX to not share stack
//! allocated values but instead use heap allocated values and move them to
//! whatever wants to access them.
//!
//! This library is focused on the async/await use-case of such a buffer,
//! a similar library like this could be written for blocking use cases.
//!
//! This library is `#[no_std]` compatible, it doesn't need alloc for any
//! functionality. But if you use it in a std context enabling the `std`
//! feature can lead to slightly better performance in some (edge) cases.
//!
//! # Example
//!
//! TODO
//!
//! # Usage-Recommendation
//!
//! - You can pass around a `&mut StackValueHandle` which is in many situations the easiest thing
//!   to do. Alternatively you can treat it similar to a `&mut Value` reference, but as there
//!   are no automatic re-borrows for custom types you will need to use the [`StackValueHandle.reborrow()`]
//!   method. The benefit of the later is that it's easier to write methods for it as you only have a
//!   single covariant lifetime in the [`StackValueHandle`] instead of having a covariant lifetime
//!   in the `&mut` and a invariant lifetime in the [`StackValueHandle`]
//!
//! - Given that there is currently no async destructor it is recommended
//!   to do a `buffer.completion().await` (or `cancellation()`) at the
//!   end of the stack frame you created the anchor on (e.g. at the
//!   end of the function you called `ra_buffer!(..)`) to
//!   prevent any unnecessary blocking during drop in the case where
//!   there is still a ongoing operation.
//!
//! # How it works
//!
//! The anchor is used to "anchor" the value to the stack. While a simple
//! implementation would contains the value itself, this one doesn't do
//! that to be able to support coercion to unsized values e.g. a `[u8]`
//! buffer. Drawback of this is that the anchor has a `unsafe` constructor.
//! But if you use the macro to create it you won't need to bother with
//! this detail.
//!
//! The idea of the [`StackAnchor`] is that on [`StackAnchor.drop()`] it will
//! *wait* for the completion of the external operation only returning from drop
//! the stack is no longer externally accessed (this includes return by panic).
//! (More details further below.)
//!
//! Semantically it will `Send` a `&mut` ref of the value to whatever executes
//! the operation and then when that thing signals the [`OperationInteraction`]
//! instance that the operation it completed the `&mut` is send back/discarded.
//! (Note that signaling the completion is inherently unsafe as it must guarantee
//! that there are no more references to the stack value, but how exactly it is
//! done is a implementation detail of the [`OperationInteraction`] instance.)
//!
//! To make sure the anchor containing the stack value is not moved/discarded
//! the [`StackAnchor`] is pinned to the stack using the [`StackValueHandle`]
//! which also makes handling lifetimes a bit easier then a raw `Pin` and works
//! around some rust limitations wrt. self-referencing and mutability.
//! (The externally accessed stack value does not need to be pinned.)
//!
//! Pin gives following [drop guarantee](https://doc.rust-lang.org/std/pin/index.html#drop-guarantee):
//!
//! > its memory will not get invalidated or repurposed from the moment it gets pinned until when drop is called.
//!
//!
//! Given the `unsafe` contract of the [`StackAnchor`] which is normally fulfilled by
//! placing the value just above the anchor on the (async) stack we have the guarantee
//! that either [`StackAnchor.drop()`] is called or the stack is leaked but never ever
//! will the memory of the stack be repurposed without calling drop on the anchor.
//! In turn [`StackAnchor.drop()`] only returns once the external stack access ended.
//!
//! A normal stack can not be leaked so drop will always be called as long as the thread
//! doesn't hang or the process aborts. On the other hand "async" stack can leak as due
//! to the async transformation values might be in the generator instead of the stack.
//! But generates only can contain a [`StackAnchor`] if they started to run in which
//! case they are pinned in which case the drop guarantee still applies to the anchor.
//! (You can leak boxed generators, but you can't leak stack pinned generators without
//! unsound unsafe code.)
//!
//! As already mentions [`StackAnchor.drop()`] will not return until the external operation
//! completes. For this we provide a way to properly await completion. Both in a sync way
//! for `drop` and a async way for methods like [`StackValueHandle.completion()`].
//!
//! One **major** drawback of this approach is that if the external operation hangs the
//! future might not only hang in a async way (e.g. on `operation.completion().await`) but
//! also on a sync way during drop if no method like `completion()` is awaited. **This
//! includes situations where a future is dropped to "cancel" it!**.
//!
//! The [`StackAnchor`] is generic over the way how it polls for completion by storing
//! a instance of [`OperationInteraction`] **in** the [`StackAnchor`] referred to by the
//! [`StackValueHandle`].
//!
//! This allows all kinds of custom schemas for polling for completion, furthermore uses
//! some trick to *soundly* allow giving out pointers to the [`OperationInteraction`]
//! instance stored in the [`StackAnchor`] (which are valid until the operation ended).
//! This means that no `alloca` based types like `Arc` are needed to store the state
//! of the completion of the operation. I.e. the [`OperationInteraction`] instance
//! itself is externally accessible (but is limited to `Sized` types).
//!
//! One possible way to implement [`OperationInteraction`] in a lock-free way is
//! provided in the `op_int_utils::atomic_state` module. Operations API's using
//! this library can very often either use it directly or wrap it in their own impl.
//!
//!
//! # More details about how it works
//!
//! ## Terminology
//!
//! ### *external accessed stack value*
//!
//! A value on the stack which in case of an async stack is accessed from
//! outside the task (future) or in case of a sync stack is accessed from
//! outside of the thread.
//!
//! The access might be by a different task/thread or something else like
//! the OS kernel or a DMA controller.
//!
//! ### *(external) operation*
//!
//! This libraries uses the term operation mainly to refers
//! to external operations done on the stack value of the [`StackAnchor`].
//! What that means is that something which is not the current async task
//! or thread (in a sync context) is *potentially* "somehow" accessing the
//! value which could or could not be ina way which mutates the value. For
//! example a DMA controller reading/writing from/to an stack allocated
//! buffer would be an ongoing operation.
//!
//! ### *stack*/*stack frame*
//!
//! A bit of memory allocated with a function call and
//! freed with the end of the function call. Rust guarantees that all
//! values stored on the stack are dropped before the stack is freed.
//! There are no exceptions, but some contains might prevent their inner
//! values from being dropped like e.g. `ManuallyDrop` or a `Rc` with
//! a reference cycle.
//!
//! ### *async stack*/*async stack frame*
//!
//! The stack frame of an `async` function.
//! Due to the async transformation the stack of an async function doesn't
//! exactly behave like a "normal" stack mainly:
//!
//! - Values which will not life across a `.await` will be placed on the normal stack.
//! - Values which will life across a `.await` boundary will instead be placed in the
//!   pinned generator of the given async function.
//!
//! Still due to values only being placed in the generator when it pinned and the
//! generate drop guarantees this behaves like a sync stack wrt. running the
//! async function and awaiting other async functions in it.
//!
//! The only practical difference is that while a normal stack frame can't really
//! be leaked (without permanent hanging a thread) a async stack frame can be leaked
//! if and only if the generator is allocated in `'static` memory (the heap or
//! a `static mut` or a `static` with interior mutability).
//!
//! This means that we still can rely on the memory not being invalidated or re-purposed
//! (without unsound unsafe code). Which means even through a future can be leaked it
//! can only be leaked in a way safe for our purpose. (As we only rely on the memory not
//! being invalidated or re-purposed).
//!
//! This also means futures which are pinned to the stack can not be leaked! Functions
//! like `block_on` without `Send` bound normally pin and poll the future on the
//! stack.
//!
//! ## How this library is safe?
//!
//! This library is safe as long as we can make sure that the stack value:
//!
//! 1. Is valid during the whole time it's used "externally".
//! 3. Can not get any kind of reference to the value of which a
//!    `&mut` ref is send to be externally accessed during the time
//!    it is potentially externally accessed (i.e. until the operation
//!    ends).
//! 2. Handles `Sync`/`Send` and aliasing correctly.
//!
//! By making sure that the memory of the value is only invalidated or
//! re-purposed if the anchor was dropped and by making sure the anchors
//! `drop()` method only returns once the external operation ended we can
//! make sure it's valid fro the whole time it's used.
//!
//! By checking for an ongoing operation before handing out any reference
//! we can make sure we don't locally access it while having handed out
//! a `&mut` externally.
//!
//! Through the unsafe contracts of [`StackAnchor.new_unchecked()`] and
//! [`StackValueHandle.new_unchecked()`] make sure that you can't move the
//! anchor to a different (async) stack then the value (the macro so safely
//! create the anchor and handle shadows the anchor making sure you can't directly
//! access it and in turn you can't move it around).
//!
//! Lastly this library uses some tricks to handle aliasing and external access
//! to the [`OperationInteraction`] as described further below. (The [`StackValueHandle`]
//! contains only a `&`-ref + interior mutability even through it behaves like a `&mut`-ref
//! this is necessary to prevent accidental no-aliasing guarantees to the [`OperationInteraction`]
//! while it's potentially aliased, this is currently necessary due to a limitation of rust).
//!
//! ## Why `StackValueHandle` instead of a `Pin`.
//!
//! There are two reasons for it:
//!
//! - Usability(1): We can implement the necessary methods on the handle, which means we
//!   can have methods based on `&mut self` and `&self` taking advantage of the borrow
//!   checker *and* providing better UX. (You still can directly reborrow the handle using
//!   [`StackValueHandle.reborrow()`] in the same way you could reborrow a `Pin` using
//!   [`Pin.as_mut()`]).
//!
//! - Usability(2): Instead of having two lifetimes which need to be handled correctly we
//!   only have one.I.e. `Pin<&'covar mut StackAnchor<'invar, V, OpInt>>` vs.
//!   `StackValueHandle<'covar, V, OpInt>`. (which is safe as the handle roughly represents a
//!   `&mut`-ref to the underlying buffer with additional functionality attached to it).
//!
//! - Rust limitations as mentioned in [Issue #63818](https://github.com/rust-lang/rust/issues/63818).
//!   This forces us the use a `Pin<&StackAnchor<..>>` and interior mutability instead of
//!   an `Pin<&mut StackAnchor>` but the handle should behave like a `&mut Buffer` wrt. the
//!   lifetime variance/re-borrowing. I.e. there should only be one (not borrowed) `&mut` to
//!   the handle at any point in time (for better ease of use, not safety).
//!   As such we wrapped pinned reference in a custom handle type.
//!
//! ## How is it safe to drop the `OperationHandle`?
//!
//! The [`OperationHandle`] is just a wrapper around a [`StackValueHandle`] which knowns
//! that there is a ongoing operation and can as such provide nicer API's for e.g. `.completion().await`.
//!
//! This means it's de-facto just like a reborrow of an `&mut` which always has no (runtime)
//! effect on being dropped. So this is always safe.
//!
//! ## How do we make sure that an operation can't override another?
//!
//! The method used by API's which start operations which this buffer fails if
//! there is a ongoing operations.
//!
//! It's generally recommended that such methods await `completion` or `cancellation`
//! of any ongoing operation first before starting a new one. Doing so while there is
//! no ongoing operation is basically `Noop` (as single `option.is_some()` call).
//!
//! ## How can we access the value safely outside of operations?
//!
//! There are methods to access the buffer once the operation ended (e.g. [`StackValueHandle.access_value_after_completion()`])
//! and methods to access the buffer failing if there is a ongoing operation (e.g. [`StackValueHandle.try_access_value())`]).
//!
//! ## Guarantees for the value passed to `StackAnchor.new_unchecked()`
//!
//! Before we slightly glossed over the guarantees a buffer passed in to
//! [`StackAnchor.new_unchecked()`] must give, *because it's strongly
//! recommended to always place it direct above the anchor on the same
//! stack*.
//!
//! The unsafe-contract rule is:
//!
//! - You can pass any value in where you guarantee that it only can be re-purposed
//!   after the anchor is dropped and that if the anchor is leaked the value is
//!   guaranteed to be leaked, too.
//!
//! But this can be tricky to get right for anything but the most simple use
//! case where you place the value directly on the stack above the anchor which
//! then is directly pinned there.
//!
//! For more complex use case consider following rules:
//!
//! - You don't need to place the value directly on the stack, placing
//!   a unique owner of it is enough for this. This can be e.g. a `Box<[V]>`,
//!   `Vec<V>` or even a `MutexGuard<[V]>` it's only important that it unique owns
//!   the value. We do not give out any drop guarantees for the value anyway,
//!   as we semantically "send" a `&mut`-ref to the value to the operation and
//!   receive it back/discard it once it's completed.
//!
//! - The value owner isn't required to be placed on the same stack frame. But it
//!   must be placed on the same or a parent stack frame and in the same or a parent
//!   future. If and only if the future is pinned onto a normal stack, then the "same or
//!   parent stack frame" rule extends from inside the async stack to the outside as the
//!   async stack (or at leas the relevant parts) are part of the "normal" stack
//!   on the outside they are placed on.
//!
//! This means you *could* have something crazy like:
//!
//! ```ignore
//! // sync stack frame
//!
//! // reuseable_buffer ~: Arc<Mutext<[u8]>>
//! let mut buffer = reusable_buffer.lock();
//! let future = async {
//!     // Safe:
//!     // 1. We do guarantee buffer to be on the same stack frame
//!     //    (due to how we pin the future to the outside stack on which also the buffer is).
//!     // 2. We shadow and pin it afterwards
//!     let mut buffer = unsafe { StackAnchor::<_, DMAInteraction>::new_unchecked(&mut buffer); };
//!     // Safe: For the same reasons `pin_utils::pin_mut!` is safe.
//!     let mut buffer = unsafe { StackValueHandle::new_unchecked(&mut buffer) };
//!     do_some_dma_magic(buffer.reborrow()).await;
//!     // make sure we properly await any pending operation aboves method might not have
//!     // awaited the completion on, so that we don't block on drop (through that kinda doesn't
//!     // matter in this example).
//!     buffer.completion().await
//! };
//! pin_mut!(future)
//! let result = block_on_pin(future);
//! ```
//!
//! While aboves example is sound, **it's strongly recommended to not do so**. As it's very
//! prone to introducing unsafe-contract breaches. E.g. just changing the last two lines to
//! a `block_on(future)` which doesn't stack pin the future would brake the unsafe-contract
//! as it no longer guarantees that the value is on the same or an parent stack of the anchor.
//!
//!
//! ## Implementing API's to start operations on the stack value
//!
//! This section contains some information for API implementers which do want to use this library.
//!
//! - The [`StackValueHandle.try_register_new_operation()`] method can be used to register a new operation.
//!
//! - A operation must only start *after* [`StackValueHandle.try_register_new_operation()`] returned successfully.
//!
//! - Semantically a `&mut V` reference to the value is passed to whatever executes the operation. Furthermore
//!   the `&mut`-ref (and any reference based on it) is guaranteed to be discarded when the operation is completed.
//!   Be aware that `&mut V` is `Send` if `V` is `Send`. Which means the value must be `Send` but doesn't need to
//!   be `Sync`.
//!
//! - To make it easier to build operations the [`StackValueHandle.get_ptr()`] method
//!   can be called *before* starting a new operation. **But the returned ptr MUST NOT be dereferenced before
//!   the operation starts.**. Even if you just created a reference but don't use it it's already a violation
//!   of the unsafe contract (this is necessary due to how compliers treat references wrt. optimizations).
//!
//! - The `OperationInteraction` instance is a arbitrary `Sized` type which implements `OperationInteraction`
//!   and as such is used to poll-await/sync-await completion of the operation and/or notify that the operation
//!   should be canceled (if supported by the operation, else requesting cancellation is a no-op).
//!
//! - The passed in [`OperationInteraction`] instance is semantically pinned, this means it will not be moved
//!   until it's dropped. Furthermore it is guaranteed that [`OperationInteraction.make_sure_operation_ended()`]
//!   is called before dropping it. Lastly it's guaranteed that there is no no-aliase constraint on the instance.
//!   **This means you can pass a `&`-ref  like reference (i.e.`*const` ptr) to the  [`OperationInteraction`]
//!   instance into whatever executes the operation as long as you make sure no `&`-ref exists after the operation
//!   completed. This works in the same way the `&mut V` ref is passed to whatever does the operations. Combining
//!   this with `Sync`/thread-safe interior mutability allows you to store state of the operation in the anchor.
//!   An example for this is the [`crate::op_int_utils::atomic_state::Anchor`]. This can be used to pass back
//!   a result from the operations, notify completion and directly cleanup the operations and passing `Waker`
//!   which needs waking to the executor of the operation (or e.g. an interrupt handle called once the operation
//!   completes).
//!
//! - The [`StackAnchor.operation_interaction()`] method can be used to get a pinned borrow to the current
//!   operation interaction. This is useful to setup/start the operation after having already registered it.
//!   It's also the only way to get a pointer to it's pinned memory location.
//!
//! - If a pointer (in-)to the [`OperationInteraction`] is passed to whatever executes the operation then following (slightly
//!   redundant) rules MUST be uphold to make it safe:
//!   - Only during the operation can the pointer be dereferenced, only during that time can
//!     a reference based on that pointer exist (even if not used).
//!   - While a reference base on that pointer exists, [`OperationInteraction.make_sure_operation_ended()`]
//!     MUST NOT return. Even if it's guaranteed that the reference is not used anymore.
//!   - After [`OperationInteraction.make_sure_operation_ended()`] returned the pointer must no longer be dereferenced
//!     at all, even if the resulting reference is not used. It's strongly recommended to discard the pointer once the
//!     operation concludes before making the completion public.
//!   - There MUST NOT be a race between the completion of the operation becoming public (`make_sure_operation_ended`
//!     potentially returning) and references being discards/the pointer no longer being dereferenced.
//!   - It extremely important to understand that just the possibility of  *having* a reference to the
//!     [`OperationInteraction`] instance after the completion becomes public can already trigger
//!     undefined behavior in the compiler backend and must avoided at all cost. The reason we need it that
//!     strict is because anything which allows us to act less strict is 100% unstable implementations details.
//!     (Also currently annotations like `noaliase` and `dereferenceable` are potentially used).
//!
//! - Whatever is used to implement an operation should make sure that it *does not leak it's way to notify
//!   that the operation completed*. Because if it does we have the problem that we either will have a permanently
//!   pending future or a permanently hanging `drop` method. Which both are really really bad. (This is the
//!   price for temporary handing out ownership of stack allocated values).
//!
//! - Whatever is used to implement an operation must make sure the `Result` type has the right trait bound
//!   nearly all operations happen semantically outside of the thread so nearly all operations need the
//!   result type to be `Send`.
//!
//! - [`crate::hooks::get_sync_awaiting_hook()`] provides a standard way to use a (fused) poll function inside
//!   of [`OperationInteraction.make_sure_operation_ended()`]. This allows normally writing one poll function
//!   for both the sync and async polling for completion. The `no_std` default implementation of hook will
//!   just busy poll the future for completion. If the `std` feature is enabled thread parking is used to
//!   await completion. Other platform specific methods (or diagnostics) can be used by replacing the *global*
//!   hook.
//!
//! Take a look at the `atomic_state` module and the use-case tests to look for examples how [`OperationInteraction`]
//! could be used.
//!
//! ### Handling task waking and completion awaiting.
//!
//! Busy poll for the completion of an operation is not very good and we can do better.
//!
//! Like already mentioned the [`OperationInteraction`] instance is itself pinned.
//! This means by using properly guarded interior mutability on some field of it
//! we can provide some memory shared between whatever executes the task and our-side
//! which polls on the task. Naturally we need to use some for of synchronization.
//! But a `atomic` is often good enough.
//!
//! With that we can do something *similar* to:
//!
//! - every time `poll_request_cancellation` or `poll_completion` is called we do:
//!   1. Check if the action already completed, if so return `Poll::Ready`
//!   2. If not clone the waker (`cx.waker().clone()`) and replace it as the
//!      new current waker (guarded with some atomic based spin lock or similar)
//!      - We might be able to optimize this by only cloning the new waker if it has
//!        changed.
//!      - If interrupts are involved locks might be a problem.
//!   3. we might need again check for complete and if so take the waker out again
//!      if it is still there and wake it (due to race conditions of atomics).
//!   4. return `Poll::Pending`
//!
//! - once the the operation executor finishes (e.g. a DMA complete interrupt) we do:
//!   1. set the state to "operation completed"
//!   2. take out any "current" waker if there is any and wake it.
//!
//! There are multiple ways how this can be implemented, some using locking other using
//! atomics and CAS operations. Depending on the implementation a lot of fine details
//! must be considered.
//!
//! For example a CAS based DAM operation which calls wake from a interrupt handler must
//! make sure the used scheduler can handle that.
//!
//! The [`crate::op_int_utils::atomic_state`] contains a generic lock-free (and potentially wait-free)
//! implementation of an  [`OperationInteraction`].
//!
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

pub mod hooks;
pub mod op_int_utils;
mod utils;

use crate::utils::abort_on_panic;
use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, marker::PhantomPinned, pin::Pin, ptr, task::Context, task::Poll};

/// Trait for type allowing interaction with an ongoing operation
///
/// # Unsafe-Contract
///
/// The implementor MUST guarantee that after [`OperationInteraction.make_sure_operation_ended()`]
/// the operation did end and the value from the anchor is no longer accessed in ANY way
/// by the operation. Neither must the operation have any form of `&`/`&mut` ref to it.
///
/// The same is true for any reference to the [`OperationInteraction`] which was handed out.
///
/// See the method documentation of [`OperationInteraction.make_sure_operation_ended()`] for more
/// details.
pub unsafe trait OperationInteraction {
    /// Type of which value is returned on completion.
    ///
    /// A typical type would be something like `Result { Succeeded, Failed }`. Through
    /// other more complex values can be returned, too.
    ///
    /// Note that most operations happen semantically outside of the thread, so in
    /// most probably all cases `Result` should be `Send`
    type Result;

    /// This method must only return when the operation ended.
    ///
    /// - This method is always called before cleaning up an operation.
    ///
    /// - It is guaranteed to be called after [`OperationInteraction.poll_completion()`] returns `Ready`.
    ///
    /// - It is guaranteed to be called before this [`OperationInteraction`] instance is dropped.
    ///
    /// - It is also always called when the anchor is dropped and there is a operation which has
    ///   not yet been cleaned up.
    ///
    /// - If possible this method should try to cancel any ongoing operation so that it can
    ///   return as fast as possible.
    ///
    /// - This method will be called both in sync and async code. It MUST NOT depend on environmental
    ///   context provided by an async runtime/executor/scheduler or similar.
    ///
    /// - The library guarantees to only call this method once. But [`StackValueHandle.operation_interaction()`]
    ///   can always be used to get a pinned reference to the operation interaction and as such this method can
    ///   theoretically be called twice. So it is okay to panic or abort if this is called twice, as this should
    ///   not happen normally.
    ///
    /// # Panic = Abort
    ///
    /// Due to safety reason a panic in a call to this method will cause an
    /// abort if the method was called by this library. This is necessary as
    /// we can no longer be sure that the buffer is accessible but
    /// as the buffer might be on the (real) stack we can also not leak it.
    ///
    /// # Safety
    ///
    /// A implementor of this method must assure on a rust safety level that
    /// once this method returns the operation ended and there is no longer
    /// ANY access from it into the buffer. **Be aware that this includes
    /// return by panic.**
    ///
    /// As such if you for whatever reason can no longer be sure that this
    /// is the case you can only either hang the thread or abort the program.
    /// Because of this this library is only nice for cases where we can make
    /// sure a operation either completed or was canceled.
    ///
    /// Note that except on `Drop` of the anchor, `poll_complete` will likely
    /// be polled before this method is called so if you can no longer say
    /// if it completed or not it's can be better to hang the future and with
    /// that permanently leak the buffer instead of hanging the thread.
    fn make_sure_operation_ended(self: Pin<&Self>);

    /// Notifies the operation that it should be canceled.
    ///
    /// Once a future using this as poll completes it means that
    /// the request for cancellation has been received be the
    /// operation. It *does not* mean has ended through cancellation.
    ///
    /// For operations which do not support cancellation this has a
    /// default implementation which instantly completes. Even operations
    /// which support cancellation can always return `Ready(())` before
    /// the operation is canceled/starts cancelling.
    ///
    /// # Extended poll guarantees.
    ///
    /// A implementor must guarantee that polling this *even after
    /// poll returned `Ready`* doesn't panic. Wrt. this this differs
    /// from a classical poll.
    ///
    fn poll_request_cancellation(self: Pin<&Self>, _cx: &mut Context) -> Poll<()> {
        Poll::Ready(())
    }

    /// Wrapped in a future this resolves once the I/O operation completed.
    ///
    /// The implementation SHOULD make sure that if `Ready` is returned the
    /// operation actually did complete. But in difference to
    /// `make_sure_operation_ended` this is not on a rust safety level as
    /// `make_sure_operation_ended` will *always* be called before allowing
    /// re-using the buffer for a new operation.
    ///
    /// # Implementor Warning (detached)
    ///
    /// The completion of the operation should *never* depend on this method
    /// being polled. Because:
    ///
    /// 1. We might never poll it (in case of `Drop`'ing the anchor).
    /// 2. We might only poll it once we hope/expect it to be roughly done.
    ///
    /// So polling this should never drive the operation to completion,
    /// only check for it to be completed.
    ///
    /// # Implementor Warning (Sync)
    ///
    /// While this library will not call `poll_completion` after it returned
    /// ready it should be noted that by using [`StackValueHandle.operation_interaction()`]
    /// any user of this library can always get a `Pin<&>`-ref to the instance,
    /// which could be used to all `poll_complete` after it returned `Ready`.
    ///
    /// Furthermore if this instance is `Sync` it could be shared to other threads
    /// and polled parallel from there. Therefore it's recommended to make the
    /// [`OperationInteraction`] instance `!Sync`. To still allow sharing it with
    /// the thing executing the operation it often can be a good idea to have an
    /// internal instance which is `Sync` and wrap it into a public `!Sync` type.
    ///
    /// # Wakers
    ///
    /// See the module level documentation about how to implement this in
    /// a way which is not just busy polling but used the `Context`'s `Waker`.
    ///
    /// # Poll Semantics
    ///
    /// Polling this after it returned `Ready` is allowed to panic or abort.
    fn poll_completion(self: Pin<&Self>, cx: &mut Context) -> Poll<Self::Result>;
}

//Note: I could use #[pin_project] but I have additional
//      restrictions for `operation_interaction` and need
//      to project into the option. So it's not worth it.
pub struct StackAnchor<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    /// Pointer to the underlying stack value which is guaranteed to lie on the same stack as this anchor is pinned to.
    ///
    /// To prevent any potential problems with aliasing we store a `*mut V` instead of an `&mut V` but with some small
    /// API changes wrt. `get_ptr()` we could store a `&mut V` as far as I know. But I prefer to be on the safe site
    /// here.
    value: *mut V,

    /// Combination of necessary type hints:
    ///
    /// 1. This type pseudo contains (is) a `&'a mut [V]` so we type hint that to
    ///    make sure there is no problem with variance, lifetimes or drop check.
    ///
    /// 2. Make sure it does NOT implement `Unpin`. `Unpin` is implemented for nearly everything
    ///    so if you need `!Unpin` you need to nearly always opt to explicitly hint this using
    ///    `PhantomPinned`.  The reason this type must be `!Unpin` is because in some situations
    ///    there will be something like self-references to it.
    type_hints: PhantomData<(&'a mut V, PhantomPinned)>,

    /// Type to interact with ongoing operations.
    ///
    /// # Pin guarantees
    ///
    /// - If this value in the cell is `None` it should be treated as if it's not pinned.
    ///
    /// - If the value is the cell is `Some` it MUST be treated as if pinned, to set it back
    ///   to none you must first call [`OperationInteraction.make_sure_operation_ended()`] and
    ///   then drop it *in place*.
    ///
    /// # &mut/Cell guarantees
    ///
    /// - If the value in the cell is `None` the anchor (and in turn the [`StackValueHandle`])
    ///   can freely access the cell as `&mut`, given they take a `&mut self` reference.
    ///
    /// - If the value in the cell is `Some` the cell MUST NOT be accessed with a `&mut` until
    ///   [`OperationInteraction.make_sure_operation_ended()`] was called. Then it can be accessed
    ///   mutable to drop the instance in place and set the option to `None`.
    ///
    /// - If the value in the cell is `Some` (and [`OperationInteraction.make_sure_operation_ended()`]
    ///   was not yet called) then there might be external `&`-like references to the [`OperationInteraction`]
    ///   instance. Once [`OperationInteraction.make_sure_operation_ended()`] was called that reference
    ///   MUST NOT exist anymore as then the instance is accessed through a `&mut`-ref which guarantees no
    ///   aliasing is happening. (And it's then dropped which would invalidate the ref.)
    ///
    /// - Generally to access this unsafe cell as mut you need a unique reference to it. For example having
    ///   a `&mut StackValueHandle` and knowing that there is no ongoing operation.
    ///
    operation_interaction: UnsafeCell<Option<ManuallyDrop<OpInt>>>,
}


/// # Safety
///
/// This is safe to send if we can send both a `&mut V` and the `OperationInteraction`.
///
/// While this does contains a `*mut V` which as a pointer is not send to be on the safe side the
/// pointer is not used in any way which would make it unsafe to send.
///
/// This also contains a `UnsafeCell` but this also doesn't affect if you can `Send `this.
///
/// *Warning:* You are supposed to stack pin the anchor so the only situation where it makes sense to send
/// it is when it's pinned on a async stack frame and you send that stack frame (i.e. the generator returned
/// by the future). Which is also why it needs to be send.
///
/// Note: That while it is safe to make this type `Sync` as far as I can tell it makes no sense as you are
/// supposed to only have access to it through the [`StackValueHandle`] which roughly emulates a `&mut` borrow
/// and as such there can only be one active borrow of that kind at a time making `Sync` pointless.
unsafe impl<'a, V, OpInt> Send for StackAnchor<'a, V, OpInt>
where
    V: ?Sized,
    for<'x> &'x mut V: Send,
    OpInt: OperationInteraction + Send
{}


/// # Safety
///
/// While this type does contain a unsafe cell its only allowed to be used if you have a known to be unique
/// reference to the anchor (a `&mut` which implies unique access to anchor, e.g. a `&mut StackValueHandle`)
/// **and** there is no ongoing operation.
///
/// The only reason we need to use interior mutability is because we can't have a `&mut anchor` without `noaliase`
/// constraints being implied on all fields of the anchor, which would be a problem wrt. sharing state with the
/// operation through the [`OperationInteraction`].
unsafe impl<'a, V, OpInt> Sync for StackAnchor<'a, V, OpInt>
where
    V: ?Sized,
    for<'x> &'x mut V: Sync,
    OpInt: OperationInteraction + Sync
{}



impl<'a, V, OpInt> StackAnchor<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    /// Create a new instance with given value.
    ///
    /// # Safety
    ///
    /// 1. You can pass a `&mut`-reference to any value in where you guarantee that it only
    ///   can be re-purposed after the anchor is dropped and that if the anchor is leaked
    ///   the value is guaranteed to be leaked, too.
    ///
    /// 2. You must `Pin` the anchor by using the [`StackValueHandle`] type and make sure it's
    ///    pinned in a way that `1.` is still uphold.
    ///
    /// 3. Between creating the anchor and pinning it through a [`StackValueHandle`] you must guarantee
    ///    not to move it in any way which would invalidate point 1.
    ///
    /// It **very strongly** recommended always do following:
    ///
    /// 1. Have the value on the stack directly above the anchor.
    /// 2. `Pin` the anchor using [`StackValueHandle::new_unchecked()`] to the stack immediately after
    ///    constructing it
    /// 3. Use the `StackValueHandle` to shadow the anchor.
    ///
    /// This is the most simple way to guarantee the unsafe contract is uphold.
    ///
    /// See module level documentation for more details.
    pub unsafe fn new_unchecked(value: &'a mut V) -> Self {
        StackAnchor {
            type_hints: PhantomData,
            value: value as *mut _ as *mut V,
            operation_interaction: UnsafeCell::new(None),
        }
    }
}

impl<'a, V, OpInt> Drop for StackAnchor<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    fn drop(&mut self) {
        // Safe: We are about to drop self and can guarantee it's no longer moved before drop
        let mut handle = unsafe { StackValueHandle::new_unchecked(self) };
        handle.cleanup_operation();
    }
}

/// Handle to interact with the external accessible stack anchored value.
///
/// You can treat this handle in two ways:
///
/// - As if it's a `&mut value`, using [`StackValueHandle.reborrow()`] is used for reborrowing.
/// - Passing around a `&mut handle` treating it roughly as if it is the value itself.
///
/// The handler provides a variety of mostly `&mut self` and `async` methods to access the
/// value, wait for completion of ongoing operations, request the cancellation of ongoing
/// operation and register new operations.
///
/// Most methods even such which you would normally be `&self` are `&mut self` as they need
/// `&mut` aliasing guarantees due to necessary internal book keeping.
///
/// **Normally, you don't want to create this manual but instead use the [`ea_stack_value!()`]
/// macro to create the anchor and handle in a guaranteed to be safe way.**
///
pub struct StackValueHandle<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    /// WARNING: While we have a `Pin<&Anchor>` it should be treated as
    ///          a `Pin<&mut Anchor>` wrt. to most aspects except that
    ///          'a must be covariant (the additional indirection which
    ///          introduces the lifetime in the anchor is abstracted away).
    ///
    ///          This means we must not expose the underlying `Pin` or
    ///          a [`Pin.as_ref()`] based re-borrow. A [`Pin.as_mut()`]
    ///          based re-borrow is fine.
    anchor: Pin<&'a StackAnchor<'a, V, OpInt>>,
}

/// # Safety
///
/// [`StackValueHandle`] is `Send` even through the anchor is not `Sync` as [`StackValueHandle`] acts like
/// a `&mut` even through it uses a `&`-ref (to work around aliasing problem).
///
/// This also means that while this handle will potentially use the interior mutability, but only through
/// `&mut` borrows to the handle. Similar while the handle does support reborrowing it only supports `&mut`
/// based reborrowing and is neither `Clone` nor `Copy`.
unsafe impl<'a,V,OpInt> Send for StackValueHandle<'a, V, OpInt>
where
    V: ?Sized,
    &'a mut StackAnchor<'a, V,OpInt>: Send,
    OpInt: OperationInteraction,
{}

/// # Safety
///
/// See the Send doc. If a `&mut` reference to the anchor would be `Sync` this is sync.
unsafe impl<'a,V,OpInt> Sync for StackValueHandle<'a, V, OpInt>
where
    V: ?Sized,
    &'a mut StackAnchor<'a, V,OpInt>: Sync,
    OpInt: OperationInteraction,
{}

impl<'a, V, OpInt> StackValueHandle<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    /// Create a new handle pinning the anchor to the stack.
    ///
    /// # Safety
    ///
    /// This calls [`Pin::new_unchecked()`] and inherited the unsafe contract from it.
    ///
    /// Furthermore this must be used correctly as described in the unsafe-contract from
    /// [`StackAnchor::new_unchecked()`].
    ///
    /// Lastly even through this internally uses a `&`-ref it acts as a `&mut`-ref, this
    /// means that the [`StackValueHandle`] instance must be the only way to access the anchor
    /// there must be no other `&`-refs or `&mut`-refs to the [`StackAnchor`]. (Through if there
    /// is a ongoing operation there might be references *into* the [`StackAnchor`] to or into it's
    /// [`OperationInteraction`] instance).
    ///
    /// Similar to `pin_utils::pin_mut!` it's fully safe if this is directly used after
    /// creating a [`StackAnchor`] (correctly) on the stack and we shadow the variable the anchor
    /// was created in.
    pub unsafe fn new_unchecked<'b>(anchor: &'a mut StackAnchor<'b, V, OpInt>) -> Self
    where
        'b: 'a,
    {
        // lifetime collapse (erase the longer living 'b and replace it with the
        // shorter living 'a also restore covariance, which is fine as we basically
        // semantically collapse a `&mut &mut value` into a `&mut value`, i.e. we
        // completely abstract way the fact that there is an additional indirection
        // in the anchor). If we could not do so as following we would have turned the
        // handle into a fat pointer directly pointing the both the anchor and the value!
        let anchor: &'a StackAnchor<'a, V, OpInt> = anchor;
        // Safe: because of
        //   - the guarantees given when calling this function
        //   - the guarantees given when creating the anchor
        let anchor = Pin::new_unchecked(anchor);
        StackValueHandle { anchor }
    }

    /// Re-borrows the handle in the same way you would re-borrow a `&mut T` ref.
    pub fn reborrow(&mut self) -> StackValueHandle<V, OpInt> {
        StackValueHandle {
            anchor: self.anchor,
        }
    }

    /// Returns a reference to the underlying operation interaction instance.
    ///
    /// Returns `None` if there is no ongoing operation.
    pub fn operation_interaction(&self) -> Option<Pin<&OpInt>> {
        //FIXME check if there is a problem with a overlap of the returned &OpInt and
        //Safe: If we have a `&self` we know there is no `&mut self`
        //      and such we know any `&` based access to operation interaction
        //      is safe to do. Furthermore the internal mutability of the cell is only
        //      used if the the option is `None` (in which case we don't return a reference)
        //      or the operation completed and `cleanup_operation()` is called. But `cleanup_operation`
        //      uses a `&mut self` and as such can not be called while the operation interaction
        //      is borrowed.
        let op_int = unsafe { &*self.anchor.operation_interaction.get() };
        let op_int = op_int.as_ref().map(|man_drop| {
            let op_int: &OpInt = &*man_drop;
            //Safe: operation interaction is pinned through the anchor pin
            unsafe { Pin::new_unchecked(&*op_int) }
        });
        op_int
    }

    /// Returns a mut reference to the underling value.
    ///
    /// If a operations is currently in process it first awaits the end of the operation.
    /// This will not try to cancel the operation. The result and a mut-ref to the underlying
    /// value are returned.
    ///
    /// This will not try to cancel any ongoing operation. If you need that you
    /// should await [`StackValueHandle.request_cancellation()`] before calling this
    /// method.
    ///
    pub async fn access_value_after_completion(&mut self) -> (&mut V, Option<OpInt::Result>) {
        let res = self.reborrow().completion().await;
        //SAFE: no ongoing operation means no self-references so &mut self is enough
        let value = unsafe { &mut *self.anchor.value };
        (value, res)
    }

    /// Returns a mut reference to the underlying value if there is no ongoing operation
    ///
    /// The `&mut self` borrow makes sure we can't start a new operation while we have
    /// an reference to the underlying value.
    pub fn try_access_value_mut(&mut self) -> Option<&mut V> {
        if self.has_pending_operation() {
            None
        } else {
            //SAFE: no ongoing operation means no self-references so &mut self is enough
            Some(unsafe { &mut *self.anchor.value })
        }
    }

    /// Returns a reference to the underlying value if there is no ongoing operation
    ///
    /// The `&self` borrow makes sure we can't start a new operation while we have
    /// an reference to the underlying value.
    pub fn try_access_value(&self) -> Option<&V> {
        if self.has_pending_operation() {
            None
        } else {
            //SAFE: no ongoing operation means no self-references so &self is enough
            Some(unsafe {  &*self.anchor.value })
        }
    }

    /// Return a pointer to the start of the the underlying value.
    ///
    /// # Safety
    ///
    /// You must guarantee that you only dereference the pointer after you registered
    /// an appropriate [`OperationInteraction`] with [`StackAnchor.try_register_new_operation()`].
    ///
    /// The reason why we have this unsafe method instead of returning the pointer
    /// from [`StackAnchor.try_register_new_operation()`] is because you
    /// might need it to create the `OperationInteraction` instance.
    ///
    /// The reason why this method is safe is because we store the value internally as pointer
    /// and as such can just return the pointer, even if there is an ongoing operation. Naturally
    /// this *doesn't mean* you can use the pointer while there is an unrelated ongoing operations.
    pub fn get_ptr(&self) -> *mut V {
        self.anchor.value
    }

    /// Set's a new operations interaction iff there is currently no ongoing operation.
    ///
    /// The returned [`OperationHandle`] should normally be wrapped by a type specific to
    /// the operation e.g. some `dma::Transfer` type or similar.
    ///
    /// Code using this should first do a `cancellation().await` (or `completion().await`)
    /// to make sure this won't fail.
    ///
    /// Normally you call [`StackValueHandle.get_ptr()`] beforehand to get a pointer
    /// to the underlying value and pass it into the constructor of an [`OperationInteraction`]
    /// instance you try to register. Then you register it and *only then* you use
    /// [`StackValueHandle.operation_interaction()`] to access it in a pinned version and
    /// start the operation.
    ///
    /// # Safety
    ///
    /// 1. You must only start any new operation after this method returns with ok.
    /// 2. You pass in the right `OperationInteraction` instance which guarantees
    ///    that once its `make_sure_operation_ended` method returns the operation
    ///    does no longer access the value in any form.
    ///
    pub unsafe fn try_register_new_operation(
        self,
        new_op_int: OpInt,
    ) -> Result<OperationHandle<'a, V, OpInt>, ()> {
        if self.has_pending_operation() {
            Err(())
        } else {
            // Safe:
            //  - We have a `&mut self` so we can access operation interaction as `&mut`
            //    IF there is no ongoing operation.
            //  - We know there is no ongoing operations as we checked this above.
            {
                let op_int = &mut *self.anchor.operation_interaction.get();
                debug_assert!(op_int.is_none());
                *op_int = Some(ManuallyDrop::new(new_op_int));
            }
            Ok(OperationHandle { anchor: self })
        }
    }

    /// Returns true if there is a "pending" operation.
    ///
    /// This operation might already have been completed but
    /// [`StackAnchor.cleanup_operation()`] wasn't called yet
    /// (which also means that we did neither await [`StackAnchor.completion()`] nor
    /// [`StackAnchor.cancellation()`])
    pub fn has_pending_operation(&self) -> bool {
        self.operation_interaction().is_some()
    }

    /// If there is an ongoing operation notify it to be canceled.
    ///
    /// This is normally called implicitly through the [`OperationHandle.cancellation()`]
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been discarded.
    pub async fn request_cancellation(&mut self) {
        if let Some(op_int) = self.operation_interaction() {
            futures_lite::future::poll_fn(|ctx| op_int.poll_request_cancellation(ctx)).await;
        }
    }

    /// If there is an ongoing operation await the completion of it.
    ///
    /// Returns the result if there was a ongoing operation.
    ///
    /// This will await [`OperationInteraction.poll_completion()`] and then calls
    /// [`StackValueHandle.cleanup_operation()`] which will internally call
    /// [`OperationInteraction.make_sure_operation_ended()`].
    ///
    /// This is normally called implicitly through the [`OperationHandle`]
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the [`OperationHandle`]
    /// has been discarded.
    pub async fn completion(&mut self) -> Option<OpInt::Result> {
        let mut result = None;
        if let Some(op_int) = self.operation_interaction() {
            result = Some(futures_lite::future::poll_fn(|ctx| op_int.poll_completion(ctx)).await);
        }
        //WARNING: Some other internal methods might rely on completion always calling
        //         `self.cleanup_operation()` at the end! So don't remove it it might
        //         brake the unsafe-contract.
        self.cleanup_operation();
        result
    }

    /// If there is an ongoing operation notify it to be canceled and await the completion of it.
    ///
    /// If it can not be canceled this will just wait normal completion.
    ///
    /// Returns the result if there was a ongoing operation.
    ///
    /// This will first await [`OperationInteraction.poll_request_cancellation()`],
    /// then [`OperationInteraction.poll_completion()`]
    /// and then calls [`StackValueHandle.cleanup_operation()`] which internally calls
    /// [`OperationInteraction.make_sure_operation_ended()`].
    ///
    /// This is normally called implicitly through the [`OperationHandle`]
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been discarded.
    pub async fn cancellation(&mut self) -> Option<OpInt::Result> {
        self.request_cancellation().await;
        self.completion().await
    }

    /// Cleanup any previous operation.
    ///
    /// This must be called before trying to set a new operation, but
    /// is implicitly called by `completion` and `cancellation`.
    ///
    /// This **should normally *not* be called directly** but only implicitly
    /// through `completion().await` or `cancellation().await`. Through there
    /// are some supper rare edge cases where exposing it is use-full.
    //HINT: This must be `&mut self` or else the `operation_interaction()` method
    //      and maybe others would become unsound.
    pub fn cleanup_operation(&mut self) {
        let mut needs_drop = false;
        if let Some(opt_int) = self.operation_interaction() {
            abort_on_panic(|| opt_int.make_sure_operation_ended());
            needs_drop = true;
        }

        if needs_drop {
            // Safe:
            //   1. We have a `&mut self` borrow. (Very important!)
            //   2. We called `make_sure_operation_ended`.
            //   3. We drop it in-place without moving it.
            //   4. We will override the now "cobbled" option field
            //      with `None` which is fine as we already dropped the
            //      pinned value and pin must only hold until it's dropped.
            unsafe {
                let outer_mut = &mut *self.anchor.operation_interaction.get();
                if let Some(man_drop) = outer_mut.as_mut() {
                    let op_int = &mut **man_drop as *mut OpInt;
                    ptr::drop_in_place(op_int);
                }
                ptr::write(outer_mut, None);
            }
        }
    }
}

/// Type wrapping a [`StackValueHandle`] encoding that there is currently an ongoing operation.
pub struct OperationHandle<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{
    anchor: StackValueHandle<'a, V, OpInt>,
}

impl<'a, 'r, V, OpInt> OperationHandle<'a, V, OpInt>
where
    V: ?Sized,
    OpInt: OperationInteraction,
{

    /// See [`StackValueHandle.access_value_after_completion()`]
    ///
    /// But needs a closure as we can't return a &mut [V] in a method consuming self.
    pub async fn access_value_after_completion<R>(mut self, value_access: impl FnOnce(&mut V, OpInt::Result) -> R) -> R {
        let (value, res) = self.anchor.access_value_after_completion().await;
        value_access(value, res.unwrap())
    }

    /// See [`StackValueHandle.completion()`].
    pub async fn completion(mut self) -> OpInt::Result {
        //SAFE[UNWRAP]: We unique borrow/own the StackValueHandle through this type
        //              and the type guarantees that there is an "ongoing" operation
        //              and we consume this type in this function call.
        self.anchor.completion().await.unwrap()
    }

    /// See [`StackValueHandle.cancellation()`]
    pub async fn cancellation(mut self) -> OpInt::Result {
        //SAFE[UNWRAP]: We unique borrow/own the StackValueHandle through this type
        //              and the type guarantees that there is an "ongoing" operation
        //              and we consume this type in this function call.
        self.anchor.cancellation().await.unwrap()
    }

    /// See [`StackValueHandle.request_cancellation()`]
    pub async fn request_cancellation(&mut self) {
        self.anchor.request_cancellation().await;
    }

    /// See [`StackValueHandle.operation_interaction()`]
    pub fn operation_interaction(&self) -> Option<Pin<&OpInt>> {
        self.anchor.operation_interaction()
    }
}

#[macro_export]
macro_rules! ea_stack_value {
    ($name:ident $(: $type:ty)? = $create:expr) => (
        // Make sure value is on the stack
        // For better understanding: This is similar in how it
        // works to the `pin_utils::pin_mut!` macro.
        let mut $name = $create;
        // Create the anchor, also coerce the type e.g. a `&mut [u8; 32]` => `&mut [u8]` coercion.
        // SAFE:
        // 1. We can use the value as it's directly on the stack above the anchor
        // 2. We directly pin the anchor to the stack as it's required.
        let mut $name = unsafe { $crate::StackAnchor::new_unchecked(&mut $name as _)};
        // Pin the anchor to the stack.
        // SAFE:
        // 1. Works like `pin_mut!` we shadow the same stack allocated value to prevent any non-pinned
        //    access to it.
        // 2. We didn't move the anchor at all it's still on the same stack as the value.
        let mut $name = unsafe {
            // Make sure the type is in the scope so that we can use it in the type-hint without explicit import.
            use $crate::StackValueHandle;
            let tmp $(: $type)? = StackValueHandle::new_unchecked(&mut $name);
            tmp
        };
    );
}

/// This type bundles a ptr and len into a type implementing `ReadBuffer` and `WriteBuffer`.
///
/// This can be useful if you have a DMA API which requires `'static` and uses buffer leaking
/// for safety and you want (and can) to wrap it in a way which would work with a non-static
/// [`StackValueHandle`] but you still need to pass in a `'static` buffer implementing the
/// `ReadBuffer` and `WriteBuffer` traits.
#[cfg(feature = "embedded-dma")]
pub struct UnsafeEmbeddedDmaBuffer<V> {
    ptr: *mut V,
    len: usize,
}

#[cfg(feature = "embedded-dma")]
const _: () = {
    use embedded_dma::{ReadBuffer, Word, WriteBuffer};

    unsafe impl<W>  Send for UnsafeEmbeddedDmaBuffer<W> where W: Send {}

    impl<W> UnsafeEmbeddedDmaBuffer<W> {
        /// Create a new unsafe buffer.
        ///
        /// # Unsafe-Contract
        ///
        /// You must "somehow" make sure that for the whole lifetime of this
        /// type ptr and len are valid.
        unsafe fn new(ptr: *mut W, len: usize) -> Self {
            Self { ptr, len }
        }
    }

    unsafe impl<'a, W> ReadBuffer for UnsafeEmbeddedDmaBuffer<W>
    where
        W: Word,
    {
        type Word = W;
        unsafe fn read_buffer(&self) -> (*const W, usize) {
            (self.ptr as *const W, self.len)
        }
    }

    unsafe impl<'a, W> WriteBuffer for UnsafeEmbeddedDmaBuffer<W>
    where
        W: Word,
    {
        type Word = W;
        unsafe fn write_buffer(&mut self) -> (*mut W, usize) {
            (self.ptr, self.len)
        }
    }

    impl<'a, V, OpInt> StackValueHandle<'a, [V], OpInt>
    where
        V: Word,
        OpInt: OperationInteraction,
    {
        /// Create a new unsafe buffer based on the underlying buffer, if there is no ongoing operation.
        ///
        /// # Unsafe-Contract
        ///
        /// You must "somehow" make sure that for the whole lifetime of this
        /// type ptr and len are valid.
        ///
        //TODO remove fallibility by getting length out of pointer
        // - requires: slice_ptr_len #71146
        // - alt requires: a pointer metadata/Pointee
        pub unsafe fn try_get_unsafe_embedded_dma_buffer(&self) -> Option<UnsafeEmbeddedDmaBuffer<V>> {
            if self.has_pending_operation() {
                None
            } else {
                let fat_ptr = self.get_ptr();
                //SAFE: as there is no pending operation
                let len = (&*fat_ptr).len();
                Some(UnsafeEmbeddedDmaBuffer::new(fat_ptr as *mut V, len))
            }
        }
    }
};

#[cfg(test)]
mod mock_operation;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    mod usage_patterns {
        use super::super::*;
        use crate::mock_operation::*;
        use core::mem;

        #[async_std::test]
        async fn leaked_operations_get_canceled_and_ended_before_new_operations() {
            let mi = async {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                // If we leaked the op we still can poll on the buffer directly
                let mi = call_and_leak_op(value.reborrow()).await;
                mi.assert_not_run();
                value.reborrow().completion().await;
                mi.assert_completion_run();

                // If we create a new operation while one is still running the old one is first canceled.
                let old_mi = call_and_leak_op(value.reborrow()).await;
                old_mi.assert_not_run();
                let mi = call_and_leak_op(value.reborrow()).await;
                old_mi.assert_cancellation_run();
                mi.assert_not_run();
                value.reborrow().cancellation().await;
                mi.assert_cancellation_run();

                {
                    // Doing re-borrows with as_mut() can be useful
                    // to avoid lifetime/move problems.
                    let mut buffer = value.reborrow();
                    let (op, mi) = call_op(buffer.reborrow()).await;
                    mi.assert_not_run();
                    op.completion().await;
                    mi.assert_completion_run();

                    let (op, mi) = call_op(buffer.reborrow()).await;
                    mi.assert_not_run();
                    op.cancellation().await;
                    mi.assert_cancellation_run();

                    // Buffer is just a fancy `&mut ...` so
                    // leaking it doesn't matter.
                    mem::forget(buffer);
                }

                let (mut op, mi) = call_op(value).await;
                mi.assert_not_run();
                // We just indicate cancellation but don't wait for
                // the cancellation to complete (just the notification that
                // it should cancel completed).
                op.request_cancellation().await;
                mi.assert_notify_cancel_run();

                // Now here it gets interesting:
                //  We still have a ongoing operation which might already have stopped,
                //  but which isn't cleaned up (removing the op.request_cancellation above makes
                //  no difference `op.request_cancellation` might be a noop).
                // Now with Drop we can't await completion because we don't have async Drop
                //  but we still will wait for completion but now blocking int he Drop destructor.
                //  ...
                mi
            }
            .await;
            // ...
            // So here after the drop the op has ended. (As a side note,
            // assert_cancellation/completion_run calls assert_end_op_check_run).
            mi.assert_op_ended_enforced();
        }

        async fn call_and_leak_op<'a>(value: StackValueHandle<'a, [u8], OpIntMock>) -> MockInfo {
            let (op, mock_info) = mock_operation(value).await;
            mem::forget(op);
            mock_info
        }

        async fn call_op<'a>(
            value: StackValueHandle<'a, [u8], OpIntMock>,
        ) -> (OperationHandle<'a, [u8], OpIntMock>, MockInfo) {
            mock_operation(value).await
        }
    }

    mod StackAnchor {

        mod try_register_new_operation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn fails_if_a_operation_is_still_pending() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);

                let (_, mock) = mock_operation(value.reborrow()).await;

                let ptr = value.reborrow().get_ptr();
                let (op_int, _, new_mock) = OpIntMock::new(ptr);
                let res = unsafe { value.reborrow().try_register_new_operation(op_int) };
                assert!(res.is_err());
                mock.assert_not_run();
                new_mock.assert_not_run();
            }

            #[async_std::test]
            async fn sets_the_operation() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let ptr= value.reborrow().get_ptr();
                let (op_int, _, mock) = OpIntMock::new(ptr);
                let res = unsafe { value.reborrow().try_register_new_operation(op_int) };
                assert!(res.is_ok());
                assert!(value.reborrow().has_pending_operation());
                mock.assert_not_run();
            }
        }

        mod cleanup_operation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn does_not_change_anything_if_there_was_no_pending_operation() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);

                let ptr = value.reborrow().get_ptr();
                let has_op = value.reborrow().has_pending_operation();

                value.reborrow().cleanup_operation();

                let ptr2 = value.reborrow().get_ptr();
                let has_op2 = value.reborrow().has_pending_operation();

                assert_eq!(ptr, ptr2);
                assert_eq!(has_op, has_op2);
            }

            #[async_std::test]
            async fn does_make_sure_the_operation_completed_without_moving() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_op, mock) = mock_operation(value.reborrow()).await;
                let op_int_addr = value
                    .operation_interaction()
                    .map(|pin| pin.get_ref() as *const _)
                    .unwrap();
                value.reborrow().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
            }

            #[async_std::test]
            async fn does_drop_the_operation_in_place() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_op, mock) = mock_operation(value.reborrow()).await;
                let op_int_addr = value
                    .reborrow()
                    .operation_interaction()
                    .map(|pin| pin.get_ref() as *const _)
                    .unwrap();
                value.reborrow().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
                mock.assert_was_dropped();
            }
        }

        mod get_buffer_ptr_and_len {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn return_the_right_pointer_and_len() {
                let mut buffer = [0u8; 32];
                let buffer = &mut buffer;
                let buff_ptr = buffer as *mut _;

                let mut anchor = unsafe { StackAnchor::<[u8], OpIntMock>::new_unchecked(buffer as _) };
                let anchor = unsafe { StackValueHandle::new_unchecked(&mut anchor) };

                let ptr = anchor.get_ptr();
                assert_eq!(ptr, buff_ptr);
            }
        }

        mod has_pending_operation {
            use super::super::super::*;
            use crate::{mock_operation::*, utils::not};

            #[async_std::test]
            async fn returns_true_if_there_is_a_not_cleaned_up_operation() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                assert!(not(value.reborrow().has_pending_operation()));
                let (op, _mock) = mock_operation(value.reborrow()).await;
                assert!(op.anchor.has_pending_operation());
                op.cancellation().await;
                assert!(not(value.reborrow().has_pending_operation()));
            }
        }

        mod request_cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn awaits_the_poll_request_cancellation_function_on_the_op_int_instance() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.request_cancellation().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod completion {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_completion() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.completion().await;
                mock.assert_completion_run();
            }

            #[async_std::test]
            async fn makes_sure_to_make_sure_operation_actually_did_end() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.completion().await;
                mock.assert_completion_run();
                mock.assert_op_ended_enforced()
            }

            #[async_std::test]
            async fn makes_sure_to_clean_up_after_completion() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.completion().await;
                mock.assert_was_dropped();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation_and_complete() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.cancellation().await;
                mock.assert_cancellation_run();
            }

            #[async_std::test]
            async fn makes_sure_that_operation_ended() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                value.cancellation().await;
                mock.assert_cancellation_run();
                mock.assert_op_ended_enforced()
            }
            #[async_std::test]
            async fn makes_sure_to_clean_up() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                value.cancellation().await;
                mock.assert_was_dropped();
            }
        }

        mod buffer_mut {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn buffer_access_awaits_completion() {
                ea_stack_value!(value: StackValueHandle<[u32], OpIntMock> = [12u32; 32]);
                let (_, mock) = mock_operation(value.reborrow()).await;
                let (mut_ref, _) = value.access_value_after_completion().await;
                assert_eq!(mut_ref, &mut [12u32; 32] as &mut [u32]);
                mock.assert_completion_run();
                mock.assert_was_dropped();
            }
        }
    }

    mod OperationHandle {

        mod completion {
            use super::super::super::*;
            use crate::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_completion() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (op, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                op.completion().await;
                mock.assert_completion_run();
            }
        }

        mod request_cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (mut op, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                op.request_cancellation().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation() {
                ea_stack_value!(value: StackValueHandle<[u8], OpIntMock> = [0u8; 32]);
                let (op, mock) = mock_operation(value.reborrow()).await;
                mock.assert_not_run();
                op.cancellation().await;
                mock.assert_cancellation_run();
            }
        }
    }
}