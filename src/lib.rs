//! This library provides a way to have a *stack allocated* buffer which can be
//! use by completion based I/O like DMA, io-uring.
//!
//! To make it possible to safely use a stack allocated buffer with completion
//! base I/O some UX drawbacks are necessary compared to the classical approach
//! of "temporary leak the heap allocated buffer until the operation completes".
//!
//! This library is focused on the async/await use-case of such a buffer,
//! a similar library like this could be written for blocking use cases.
//!
//! This library is `#[no_std]` compatible, it doesn't need alloc.
//!
//! While completion based (or other "external" I/O) operations can be
//! done with this buffer for each API it's meant to be used with some
//! glue code is necessary. The main idea is that safe abstractions around
//! such API's might support this buffer from out of the box maybe even
//! use it as a fundamental component. But if not it normally can be
//! adapted to work with such APIs.
//!
//! # Example
//!
//! TODO
//!
//! # Usage-Recommendation
//!
//! - You can pass around a `&mut RABufferHandle` which is in many situations the easiest thing
//!   to do. Alternatively you can treat it similar to a `&mut Buffer` reference, but as there
//!   are no automatic re-borrows for custom types you will need to use the [`RABufferHandle.reborrow()`]
//!   method. The benefit of the later is that it's easier to write methods for it as you only have a
//!   single covariant lifetime in the [`RABufferHandle`] instead of having a covariant lifetime
//!   in the `&mut` and a invariant lifetime in the [`RABufferHandle`]
//!
//! - Given that there is currently no async destructor it is recommended
//!   to do a `buffer.completion().await` (or `cancellation()`) at the
//!   end of the stack frame you created the anchor on (e.g. at the
//!   end of the function you called `ra_buffer_anchor!(..)`) to
//!   prevent any unnecessary blocking during drop in the case where
//!   there is still a ongoing operation.
//!
//! # Principle (short/TL;DR)
//!
//! The [`RABufferHandle`] pins the [`RABufferAnchor`] to the stack.
//!
//! Due to the way `Pin` works and the unsafe contract of the buffer
//! constructor we can be assured that `Drop` of the anchor is called
//! before the memory of the buffer is potentially re-purposed (including
//! freeing it).
//!
//! We provide ways to properly await the completion of a operation.
//!
//! We always implicitly await the completion of an operation before
//! (semantically) reclaiming the ownership of the stack buffer.
//!
//! For example [`RABufferHandle.buffer_mut()`] hands out a `&mut [V]`
//! to the underlying buffer but first will await that any ongoing
//! operation completed.
//!
//! Once `Drop` is called we make sure to block until any ongoing operation
//! ended to assure that an external operation never accesses the then
//! potentially re-purposed (or just freed) memory.
//!
//! # Principle (full)
//!
//! ## Terminology
//!
//! - *stack frame*: A bit of memory allocated with a function call and
//!   freed with the end of the function call. Rust guarantees that all
//!   values stored on the stack are dropped before the stack is freed.
//!   There are no exceptions, but some contains might prevent their inner
//!   values from being dropped like e.g. `ManuallyDrop` or a custom `LeakyBox`
//!   type which acts like a box which leaks it's value on `Drop`. You can't
//!   really leak the normal stack.
//!
//! - *stack frame*: A stack frame in a "normal" function which used
//!   the "normal" stack allocation/free mechanism.
//!
//! - *async stack frame*: A stack frame a `async` function, due to the transformation
//!   done by async this isn't necessary mapping to a "normal" stack frame instead all
//!   stack allocated values which live across a (potential) `.await` call will be
//!   allocated inside of the pinned future which `async` turned the function into.
//!   This future might lay on the "normal" stack, the heap (`Pin<Box<...>>`) or it
//!   might be inside another future which awaited this future directly. In difference
//!   to the normal stack frame a *async stack frame* can be leaked, but only if it
//!   is on the heap (see below). Besides that the guarantees are roughly the same.
//!   I.e. all values are guaranteed to get dropped before the stack if freed, this
//!   is possible as dropping the future will drop all values currently inside of the
//!   pinned future.
//!
//! ## Relevant workings of Pin and async/await
//!
//! The main guarantee `Pin<&mut T>` gives you is that `T` is "pinned" in memory,
//! i.e. it will not be moved around. While this is important for our use-case
//! we also heavily rely on the "drop guarantee" `Pin` gives us. To quote:
//!
//! > for pinned data you have to maintain the invariant that its memory will not
//! > get *invalidated* or *repurposed* from the moment it gets pinned until when drop
//! > is called. Only once drop returns or panics, the memory may be reused.
//!
//! The means that you can not free the memory a value was pinned to without
//! dropping the value. Neither can you *repurpose* the memory.
//!
//! This mean that if you pin a value to a stack frame (weather async or not) you
//! must either not leak the value or leak the whole stack frame (which isn't possible
//! for normal stacks frames, but is possible for values pinned to a async stack frame
//! where the async transformation moves them into the future and the future is pinned
//! on the heap e.g. with `Pin<Box<...>>`).
//!
//! It is important that while that is a unsafe-contract which needs to be uphold by the
//! user of `Pin` futures are made so that the contract will be uphold. Or else we would
//! not be able to use them at all.
//!
//! ## How this library is safe.
//!
//! This library requires a anchor to be pinned onto a (potentially async) stack frame.
//! Furthermore the buffer the anchors anchors must uphold certain guarantees (see below),
//! which are most easily fulfilled by placing the buffer directly onto the stack above
//! the anchor.
//!
//! Furthermore it requires the anchor to only be accessible through the [`RABufferHandle`]
//! which wraps the `Pin`. (This is best done the same way `pin_utils::pin_mut!` works.)
//!
//! With that we can guarantee that either both the anchor and buffer leak together which
//! means the buffer will not be repurposed or the anchors destructor is run before there
//! is any chance of the buffer being repurposed.
//!
//! By blocking in the destructor until any operation on the buffer completes we can prevent
//! the buffers memory from being re-purposed without the operation concluding. Which in turn
//! can make it safe to use the buffer with completion based I/O like e.g. DMA.
//!
//! To explain this better lets look at following code as if produced by the
//! `ra_buffer_anchor!(buffer = [0u8; 32] of DMAInteraction)` code:
//!
//! ```no_run
//! # use core::{pin::Pin, task::{Context, Poll}};
//! # use remote_accessed_buffer::*;
//! # struct DMAInteraction;
//! # unsafe impl OperationInteraction for DMAInteraction {
//! #   type Result = ();
//! #   fn make_sure_operation_ended(self: Pin<&Self>) { todo!() }
//! #   fn poll_request_cancellation(self: Pin<&Self>, cx: &mut Context) -> Poll<()> { todo!() }
//! #   fn poll_completion(self: Pin<&Self>, cx: &mut Context) -> Poll<()> { todo!() }
//! # }
//! let mut buffer = [0u8; 32];
//! // SAFE:
//! // 1. We can use the buffer as it's directly on the stack above the anchor
//! // 2. We directly pin the anchor to the stack as it's required.
//! let mut buffer = unsafe { RABufferAnchor::<_, DMAInteraction>::new_unchecked(&mut buffer) };
//! // SAFE:
//! // 1. Works like `pin_mut!` we shadow the same stack allocated buffer to prevent any non-pinned
//! //    access to it. (The pin is is wrapped in the `RABufferHandle`.)
//! let mut buffer = unsafe { RABufferHandle::new_unchecked(&mut buffer) };
//! ```
//!
//! Here by having the array buffer on the same stack as the anchor and making the buffer
//! out-life the anchor (it's on the stack "above" the buffer) we know that either we will
//! leak both the buffer and the anchor (which is ok) or the anchor will be dropped be the
//! buffer can maybe be accessed again (in the example case it can not as we shadow
//! the buffer, too).
//!
//! Semantically seen the anchor takes ownership of the buffer until it's dropped.
//!
//! Furthermore by shadowing the anchor with it's pin we make sure it's impossible to
//! access the anchor through anything but the `Pin`. If we would not have shadowed it
//! we would need to manually guarantee that it's not accessed which is just really annoying
//! to do and error prone. Note that this aspects works exactly the same as `pin_mut!`.
//! I literally could have replaced the last line with `pin_mut!(buffer)` having one
//! unsafe line less in my code (but instead in `pin_mut!`). But for demonstration
//! purpose and readability writing it out by hand is better, here.
//!
//! Now one think which should be mentioned is that needing to wait for completion in
//! `Drop::drop` is not the best thing to do. We can not get around it but having something
//! like a `AsyncDropPreparation` trait which automatically does something like a async
//! drop before the drop would be supper helpful (a `AsyncDrop` which replaces `Drop` can't
//! really work as far as I know).
//!
//! Because of this it's strongly recommended to make sure that under normal circumstances
//! `buffer.completion().await` or `buffer.cancellation().await` is run before dropping it.
//!
//! ## Why `RABufferHandle` instead of a `Pin`.
//!
//! There are two reasons for it:
//!
//! - Usability(1): We can implement the necessary methods on the handle, which means we
//!   can have methods based on `&mut self` and `&self` taking advantage of the borrow
//!   checker _and_ providing better UX. (You still can directly reborrow the handle using
//!   [`RABufferHandle.reborrow()`] in the same way you could reborrow a `Pin` using
//!   [`Pin.as_mut()`]).
//!
//! - Usability(2): Instead of having two lifetimes which need to be handled correctly we
//!   only have one.I.e. `Pin<&'covar mut RABufferAnchor<'invar, V, OpInt>>` vs.
//!   `RABufferHandle<'covar, V, OpInt>`.
//!
//! - Rust limitations as mentioned in [Issue #63818](https://github.com/rust-lang/rust/issues/63818).
//!   This forces us the use a `Pin<&RABufferAnchor<..>>` and interiour mutability instead of
//!   an `Pin<&mut RABufferAnchor>` but the handle should behave like a `&mut Buffer` wrt. the
//!   lifetime variance/re-borrowing. I.e. there should only be one (not borrowed) `&mut` to
//!   the handle at any point in time (for better ease of use, not safety).
//!   As such we wrapped pinned reference in a custom handle type.
//!
//! ## How is it safe to drop the `OperationHandle`?
//!
//! If a operation started a `OperationInteraction` instance is registered on the buffer.
//! This instance can be used to check if the operation ended.
//!
//! This instance is *not* placed in the `OperationHandle` returned. Instead it is placed
//! "upstream" in the pinned anchor. This mean we can guarantee to have access to it until
//! we explicitly remove it because we realized the operation ended.
//!
//! The `OperationHandle` just wraps the used `RABufferHandle` to have a
//! way to encode that fact that there is a ongoing operation in the type system.
//!
//! You can always drop the `OperationHandle` handle which will literally have no affect
//! on the operation itself.
//!
//! The reason for this is that we anyway can access the `RABufferHandle`
//! while a operation is ongoing by using re-borrows + (safe) leaking. In the end awaiting
//! completion/cancellation on the `OperationHandle` just forwards it the the same named
//! methods on `RABufferHandle`!
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
//! ## How can we access the buffer safely outside of operations?
//!
//! There are two methods [`RABufferHandle.buffer_mut()`] and [`RABufferAnchor.buffer_ref()`]
//! which return a `&mut [V]`/`&[V]` after awaiting completion of any still ongoing
//! operation.
//!
//! ## Guarantees for the passed in buffer
//!
//! Before we slightly glossed over the guarantees a buffer passed in to
//! [`RABufferAnchor.new_unchecked()`] must give, *because it's strongly
//! recommended to always place it direct above the anchor on the same
//! stack*.
//!
//! The unsafe-contract rule is:
//!
//! - You can pass any buffer in where you guarantee that it only can be re-purposed
//!   after the anchor is dropped and that if the anchor is leaked the buffer is
//!   guaranteed to be leaked, too.
//!
//! But this can be tricky to get right for anything but the most simple use
//! case where you place the buffer directly on the stack above the anchor which
//! then is directly pinned there.
//!
//! For more complex use case consider following rules:
//!
//! - You don't need to place the buffer directly on the stack, placing
//!   a owner of it is enough for this. This can be e.g. a `Box<[V]>`,
//!   `Vec<V>` or even a `MutexGuard<[V]>` it's only important that it owns
//!   the buffer. We do not give out any drop guarantees for the buffer anyway,
//!   as we semantically "send" the buffer to the operation and receive it back
//!   once it's completed.
//!
//! - The buffer owner isn't required to be placed on the same stack frame. But it
//!   must be placed on the same or a parent stack frame and in the same or a parent
//!   future. If and only if the future is pinned onto a normal stack, then the "same or
//!   parent stack frame" rule extends from inside the async stack to the outside as the
//!   async stack (or at leas the relevant parts) are part of the "normal" stack
//!   on the outside they are placed on.
//!
//! This means you could have something crazy like:
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
//!     let mut buffer = unsafe { RABufferAnchor::<_, DMAInteraction>::new_unchecked(&mut buffer); };
//!     // Safe: For the same reasons `pin_utils::pin_mut!` is safe.
//!     let mut buffer = unsafe { RABufferHandle::new_unchecked(&mut buffer) };
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
//! While aboves example is sound, *it's strongly recommended to not do so*. As it's very
//! prone to introducing unsafe-contract breaches. E.g. just changing the last two lines to
//! `smol::block_on(future)` would brake the unsafe-contract as it no longer guarantees that
//! the buffer is on the same or an parent stack of the anchor.
//!
//!
//! ## Implementing Operations with this buffer
//!
//! This section contains some information for API implementers which do want to use this buffer.
//!
//! - The [`RABufferHandle.try_register_new_operation()`] method can be used to register a new operation.
//!
//! - As it will fail if there is still a ongoing operation ist recommended that the method calling register
//!   does a `buffer.cancellation().await`.  Or in some special cases a `buffer..completion().await`.
//!   But generally starting a new operation should cancel ongoing operations which have
//!   not yet been completed if possible. Only the [`RABufferHandle.buffer_mut()`]
//!   method does implicitly awaiting of completion instead of cancellation as this is much better wrt.
//!   usability.
//!
//! - A operation must only start *after* [`RABufferHandle.try_register_new_operation()`] returned successfully.
//!
//! - Semantically the ownership of the buffer is passed to whatever executes the operation until it completes.
//!   This is also why the anchor stores a pointer instead of a reference to the buffer. It's best to think
//!   completion based background operations done by a DMA-controller or the OS kernel as if they are done
//!   by a different thread over which you have no control over. At least wrt. `Sync`/`Send` requirements.
//!
//! - To make it easier to build operations the [`RABufferHandle.get_buffer_ptr_and_len()`] method
//!   can be called *before* starting a new operation. **But the returned ptr MUST NOT be dereferenced before
//!   the operation starts.**. Even if you just created a reference but don't use it it's already a violation
//!   of the unsafe contract (this is necessary due to how compliers treat references wrt. optimizations).
//!
//! - The `OperationInteraction` instance is a arbitrary `Sized` type which implements `OperationInteraction`
//!   and as such is used to poll/await/sync await completion of the operation and/or notify that the operation
//!   should be canceled.
//!
//! - The passed in [`OperationInteraction`] instance is semantically pinned, this means it will not be moved
//!   until it's dropped. Furthermore it is guaranteed that [`OperationInteraction.make_sure_operation_ended()`]
//!   is called before dropping it. By combining this knowledge with interior mutability it's possible to pass
//!   pointers *into* the [`OperationInteraction`] instance to whatever does execute the operation. This is safe as
//!   similar to the buffer the [`OperationInteraction`] instance won't be dropped before the operation completes.
//!
//! - The [`RABufferAnchor.operation_interaction()`] method can be used to get a pinned borrow to the current
//!   operation interaction. This is useful to setup/start the operation after having already registered it.
//!   It's also the only way to get a pointer to it's pinned memory location.
//!
//! - Be aware that due to [rust issue #638181](https://github.com/rust-lang/rust/issues/63818) we currently
//!   must use a `Pin<&>` + interior mutability while once that issue is fixed the poll methods will switch
//!   to `Pin<&mut>` and interior stack mutability will be archived by 1st using something like a `UnsafeAliasingCell`
//!   which "punch holes" the implicit `&mut` aliasing to all fields and then combining this with a `UnsafeCell` to
//!   gain mutability again (but now potentially accessed from different thread/interrupt while there is a `&mut` borrow
//!   for polling.)
//!
//! - If a pointers (in-)to the [`OperationInteraction`] is passed to whatever executes the DMA then it always should only
//!   be to a field inside of the [`OperationInteraction`] instance but not the [`OperationInteraction`] itself. This is
//!   to make it impossible to call `poll*()` parallely and to make it easier to migrate to `UnsafeAliasingCell` or whatever
//!   the rust-language will introduce to work around issue #63818.
//!
//! - If a pointer (in-)to the [`OperationInteraction`] is passed to whatever executes the DMA then following (slightly
//!   redundant) rules MUST be uphold to make it safe:
//!   - Only during the operation can the pointer be dereferenced, only during that time can
//!     a reference based on that pointer exist (even if not used).
//!   - While there a reference base on that pointer exists [`OperationInteraction.make_sure_operation_ended()`]
//!     MUST NOT return. Even if it's guaranteed that the reference is not used anymore.
//!   - After [`OperationInteraction.make_sure_operation_ended()`] returned the pointer must no longer be dereferenced
//!     at all, even if the resulting reference is not used. It's strongly recommended to discard the pointer once the
//!     operation concludes before making the completion public.
//!   - There MUST NOT be a race between the completion of the operation becoming public (`make_sure_operation_ended`
//!     potentially returning) and references being discards/the pointer no longer being dereferenced.
//!   - It extremely important to understand that just the possibility of  *having* a reference to the
//!     [`OperationInteraction`] instance after the completion becomes public can already trigger
//!     undefined behavior in the compiler backend and must avoided at all cost.
//!
//! - Whatever is used to implement an operation should make sure that it *does not leak it's way to notify
//!   that the operation completed*. Because if it does we have the problem that we either will have a permanently
//!   pending future or a permanently hanging `drop` method. Which both are really really bad. (This is the
//!   price for temporary handing out ownership of stack allocated buffers).
//!
//! - Whatever is used to implement an operation must make sure the `Result` type has the right trait bound
//!   nearly all operations happen semantically outside of the thread so nearly all operations need the
//!   result type to be `Send`.
//!
//! Below is the **pseudo-code** of how a function starting a DMA transfer might look:
//!
//! FIXME: Test that pseudo code in a integration test.
//!
//! ```ignore
//! // call like `start_dma_operation(buffer.reborrow(), Direction::FromMemory, Periphery::FooBarDataPort).await;`
//! async fn start_dma_operation<'a, T, C >(
//!     mut buffer: RABufferHandle<'a, T::Word, DMAInteraction>,
//!     direction: Direction,
//!     channel: C, //the lifetime sucks bad time, we will fix that in newer version
//!     target: T
//! ) -> DMAOperationHandle
//! where
//!     T: dma::Target
//!     C: dma::Channel
//! {
//!     buffer.cancellation().await;
//!     let (ptr, len) = buffer.get_buffer_ptr_and_len();
//!     let interaction = DMAInteraction::new(ptr, len, channel, target);
//!     // Safe: We register the right interaction.
//!     let inner_operation_handle = unsafe { buffer.try_register_new_operation(interaction) };
//!     // SAFE(unwrap): We made sure the operation ended by polling cancellation
//!     let operation_handle = operation_handle.unwrap();
//!     let operation_handle = setup_completion_interrupt::<T,C>(operation_handle);
//!     // SAFE: We must only call `start` once before it started.
//!     unsafe { operation_handle.start() };
//!     operation_handle
//! }
//!
//! fn setup_completion_interrupt<T,C>(operation_handle: OperationHandle<T::Word, DMAInteraction>) -> DMAOperationHandle
//! where
//!     T: dma::Target,
//!     C: dma::Channel
//! {
//!     let op_int: DMAInteraction = operation_handle.operation_interaction();
//!     let completer = op_int.state_anchor.create_completer();
//!     let interrupt_data_slot = get_interrupt_data_slot_for_channel(&op_int.channel);
//!     interrupt_data_slot.set_completer(completer);
//!     DMAOperationHandle { inner: operation_handle }
//! }
//!
//! //[...]
//! let (buffer, channel, target, result) =  dma_op_handle.completion().await;
//! if let DMAResult::Completed = result {
//!     //[...]
//! }
//! //[...]
//! ```
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
//! The [`op_int_utils`] module contains some helpful utilities for implementing aboves
//! pattern using atomics in a way which should work with interrupts *if* waking a waker
//! in the used async runtime can be done from an interrupt.
//!
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

mod utils;
#[cfg(feature="op_int_utils")]
pub mod op_int_utils;

use core::{cell::UnsafeCell, marker::PhantomData, mem::ManuallyDrop, pin::Pin, ptr, task::Context, task::Poll};
use crate::utils::abort_on_panic;
/// Trait for type allowing interaction with an ongoing operation
///
/// # Unsafe-Contract
///
/// The implementor MUST guarantee that after [`OperationInteraction.make_sure_operation_ended()`]
/// the operation did end and the buffer is no longer accessed in ANY way
/// by the operation. This means there must no longer be any references
/// to the [`OperationInteraction`] from the outside (even if not used)
/// neither must pointer from the outside to the [`OperationInteraction`] instance
/// no longer be dereferenced.
///
/// See the method documentation of [`OperationInteraction.make_sure_operation_ended()`] for more
/// details.
pub unsafe trait OperationInteraction {

    /// Type of which value is returned on completion.
    ///
    /// A value which indicates which the operation failed or succeeded can returned.
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
    /// - It MUST be no problem to call this method more then once.
    ///   (FIXME: We should be able to guarantee that it's only called once by us?)
    ///
    /// # Panic = Abort
    ///
    /// Due to safety reason a panic in a call to this method will cause an
    /// abort as we can no longer be sure that the buffer is accessible but
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
    /// Note that except on `Drop` of the anchor, `poll_complete` will always
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
    /// default implementation which instantly completes.
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
    /// # Implementor Warning
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
    /// # Wakers
    ///
    /// See the module level documentation about how to implement this in
    /// a way which is not just busy polling but used the `Context`'s `Waker`.
    fn poll_completion(self: Pin<&Self>, cx: &mut Context) -> Poll<Self::Result>;
}

//Note: I could use #[pin_project] but I have additional
//      restrictions for `operation_interaction` and need
//      to project into the option. So it's not worth it.
pub struct RABufferAnchor<'a, V, OpInt>
where
    OpInt: OperationInteraction,
{

    /// We store the buffer as a tuple of a pointer to it's start and it's length.
    ///
    /// The reason for this are:
    ///
    /// 1. It's the format most (all?) operators accept as input.
    /// 2. There is currently no reliable way on stable to get then length out of
    ///    an `*mut [V]` fat pointer. And the methods returning the pointer and length
    ///    *might* be called before a pending but detached/leaked operation finished (which
    ///    is ok as long as they don't use the pointer).
    /// 3. There is a stable way to create a `&mut [V]` from the pointer to the start and
    ///    the length. (Which MUST only be used once any previous operation ended and a new
    ///    `OperationInteraction` instance for it was registered).
    ///
    buffer: (*mut V, usize),
    buffer_type_hint: PhantomData<&'a mut [V]>,

    /// Type to interact with ongoing operations.
    ///
    /// # Pin Safety (wrt. Pin<&Self>)
    ///
    /// This type doesn't use any self-referencing to this field, but
    /// [`OperationInteraction`] instances might be self-referential.
    ///
    /// If it is `None` it always can safely be replaced with `Some`.
    ///
    /// This field should never be `Some` if `Self` is not pinned.
    ///
    /// If it's `Some` we MUST treat it as if it's pinned.
    ///
    /// After [`OperationInteraction.make_sure_operation_ended()`] returned we
    /// have the guarantee that we can safely drop the [`OperationInteraction`]
    /// **in-place**, after which we can safely replace the `Some` with `None`.
    ///
    //FIXME: Reformulate:
    /// Accessing the `UnsafeCell` as `&mut `is safe *if and only if* either
    /// the operation did not yet start (it's `None`) or the
    /// operation completed, i.e. [`OperationInteraction.make_sure_operation_ended()`]
    /// was run and returned and no new operation was registered afterwards.
    ///
    /// Given that this anchor is neither `Send` nor `Sync` we can be sure that
    /// under aboves circumstances (None or completed) there is no concurrent access
    /// to the field and as such accessing it as `&mut` is safe (as already mentioned).
    ///
    //Hint: We know it's not `Sync`/`Send` due to the `*mut W` pointer and the `UnsafeCell`.
    ///
    /// So if we want to set it to none or replace it with another
    /// operation we MUST do following:
    ///
    /// 1. first calling [`OperationInteraction.make_sure_operation_ended()`] on it
    ///    **without** moving it
    /// 2. dropping it in place
    ///
    /// Failing to do so is treated as `unsafe`. This also counts for
    /// this types `Drop::drop` implementation.
    ///
    /// This allows us to not only allow the remote accessor to access
    /// the buffer but also to allow it to access (well defined) parts
    /// of the `OpInt` instance. Which e.g. could be used by an interrupt
    /// to indicate the completion of an operation or get the current
    /// waker of an future polling on `poll_completion` and waking it (
    /// assuming the async runtime implements wake in a way which can
    /// be called from an interrupt).
    operation_interaction: UnsafeCell<Option<ManuallyDrop<OpInt>>>
}


impl<'a, V, OpInt> RABufferAnchor<'a, V, OpInt>
where
    OpInt: OperationInteraction
{

    /// Create a new instance with given buffer.
    ///
    /// # Unsafe-Contract
    ///
    /// 1. You can pass any buffer in where you guarantee that it only can be re-purposed
    ///   after the anchor is dropped and that if the anchor is leaked the buffer is
    ///   guaranteed to be leaked, too.
    ///
    /// 2. You must `Pin` the anchor by using the `RABufferHandle` type and make sure it's
    ///    pinned in a way that `1.` is still uphold.
    ///
    /// It **very strongly** recommended always do following:
    ///
    /// 1. Have the buffer on the stack directly above the anchor.
    /// 2. `Pin` the anchor using [`RABufferHandle::new_unchecked()`] to the stack immediately after
    ///    constructing it
    /// 3. Use the `RABufferHandle` to shadow the anchor.
    ///
    /// This is the most simple way to guarantee the unsafe contract is uphold.
    ///
    /// See module level documentation for more details.
    pub unsafe fn new_unchecked(buffer: &'a mut [V]) -> Self {
        RABufferAnchor {
            buffer_type_hint: PhantomData,
            buffer: (buffer as *mut _ as *mut V, buffer.len()),
            operation_interaction: UnsafeCell::new(None),
        }
    }
}

impl<'a, V, OpInt> Drop for RABufferAnchor<'a, V, OpInt>
where
    OpInt: OperationInteraction
{
    fn drop(&mut self) {
        // Safe: We are about to drop self and can guarantee it's no longer moved before drop
        let mut handle = unsafe { RABufferHandle::new_unchecked(self) };
        handle.cleanup_operation();
    }
}

/// TODO TODO TODO
///
/// TODO: Variance as if &'a mut [..]
///       Which means covariance over 'a and invariance over
///       V and OpInt
///
/// TODO: Acts like a `Pin`
pub struct RABufferHandle<'a, V, OpInt>
where
    OpInt: OperationInteraction
{
    /// WARNING: While we have a `Pin<&Anchor>` it should be treated as
    ///          a `Pin<&mut Anchor>` wrt. to most aspects except that
    ///          'a must be covariant (the additional indirection which
    ///          introduces the lifetime in the anchor is abstracted away).
    ///
    ///          This means we must not expose the underlying `Pin` or
    ///          a [`Pin.as_ref()`] based re-borrow. A [`Pin.as_mut()`]
    ///          based re-borrow is fine.
    anchor: Pin<&'a RABufferAnchor<'a, V, OpInt>>
}

impl<'a, V, OpInt> RABufferHandle<'a, V, OpInt>
where
    OpInt: OperationInteraction
{

    ///
    /// # Unsafe-Contract
    ///
    /// This calls [`Pin::new_unchecked()`] and inherites the unsafe contract from it.
    ///
    /// Furthermore this must be used correctly as described in the unsafe-contract from
    /// `RABufferAnchor::new_unchecked()`.
    ///
    /// Similar to `pin_utils::pin_mut!` it's fully safe if this is directly used after
    /// creating a `RABufferAnchor` on the stack and we shadow the variable the anchor
    /// was created in.
    pub unsafe fn new_unchecked<'b>(anchor: &'a mut RABufferAnchor<'b,V,OpInt>) -> Self
    where
        'b: 'a
    {
        // lifetime collapse (erase the longer living 'b and replace it with the
        // shorter living 'a also restore covariance, which is fine as we basically
        // semantically collapse a `&mut &mut buffer` into a `&mut buffer`, i.e. we
        // completely abstract way the fact that there is an additional indirection
        // in the anchor). If we could not do so as following we would have turned the
        // handle into a fat pointer directly pointing the both the anchor and the buffer!
        let anchor: &'a RABufferAnchor<'a, V, OpInt> = anchor;
        // Safe: because of
        //   - the guarantees given when calling this function
        //   - the guarantees given when creating the anchor
        let anchor = Pin::new_unchecked(anchor);
        RABufferHandle { anchor }
    }

    /// Re-borrows the handle in the same way you would re-borrow a `&mut T` ref.
    pub fn reborrow(&mut self) -> RABufferHandle<V, OpInt> {
        RABufferHandle { anchor: self.anchor }
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
        let op_int = unsafe {
            &*self.anchor.operation_interaction.get()
        };
        let op_int = op_int.as_ref().map(|man_drop| {
            let op_int: &OpInt = &*man_drop;
            //Safe: operation interaction is pinned through the anchor pin
            unsafe { Pin::new_unchecked(&*op_int) }
        });
        op_int
    }


    /// Returns a mut reference to the underling buffer.
    ///
    /// If a operations is currently in process it first awaits the end of the operation.
    ///
    /// This will not try to cancel any ongoing operation. If you need that you
    /// should await [`RABuffer::request_cancellation()`] before calling this
    /// method.
    ///
    /// Note that there is no `buffer`/`buffer_ref` as we always need a `&mut self`
    /// anyway to access the buffer.
    pub async fn buffer_mut(&mut self) -> &mut [V] {
        self.reborrow().completion().await;
        let (ptr, len) = self.anchor.buffer;
        // Safe: We have a (pinned) &mut borrow to the anchor and we made
        //       sure it's completed (completion always calls `cleanup_operation`).
        unsafe {
            core::slice::from_raw_parts_mut(ptr, len)
        }
    }


    /// Return a pointer to the start of the the underlying buffer and it's size.
    ///
    /// # Safety
    ///
    /// You must guarantee that you only use the pointer after you installed
    /// a appropriate `OperationInteraction` with [`RABufferAnchor.try_register_new_operation()`].
    ///
    /// The reason why we have this unsafe method instead of returning the pointer
    /// from [`RABufferAnchor.try_register_new_operation()`] is because you
    /// might need it to create the `OperationInteraction` instance.
    pub fn get_buffer_ptr_and_len(&self) -> (*mut V, usize) {
        self.anchor.buffer
    }

    /// Set's a new operations interaction iff there is currently no ongoing interaction.
    ///
    /// The returned `OperationHandle` should normally be wrapped by a type specific to
    /// the operation e.g. some `dma::Transfer` type or similar.
    ///
    /// Code using this should first do a `cancellation().await` (or `completion().await`)
    /// to make sure this won't fail.
    ///
    /// # Safety
    ///
    /// 1. You must only start any new operation after this method returns.
    /// 2. You pass in the right `OperationInteraction` instance which guarantees
    ///    that once its `make_sure_operation_ended` method returns the operation
    ///    does no longer access the buffer in any form.
    pub unsafe fn try_register_new_operation(self, new_op_int: OpInt) -> Result<OperationHandle<'a, V, OpInt>, ()> {
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
    /// [`RABufferAnchor.cleanup_operation()`] wasn't called yet
    /// (which also means that we did neither await [`RABufferAnchor.completion()`] nor
    /// [`RABufferAnchor.cancellation()`])
    pub fn has_pending_operation(&self) -> bool {
        self.operation_interaction().is_some()
    }

    /// If there is an ongoing operation notify it to be canceled.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.
    pub async fn request_cancellation(&mut self) {
        if let Some(op_int) = self.operation_interaction() {
            futures_lite::future::poll_fn(|ctx| op_int.poll_request_cancellation(ctx)).await;
        }
    }

    /// If there is an ongoing operation await the completion of it.
    ///
    /// Returns the result if there was a ongoing operation.
    ///
    /// This will await `op_int.poll_completion` and then call `cleanup_operation`.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.ge).
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
    /// This will first await `op_int.poll_request_cancellation`, then `op_int.poll_completion`
    /// and then calls `cleanup_operation`.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.
    pub async fn cancellation(&mut self) -> Option<OpInt::Result> {
        self.request_cancellation().await;
        self.completion().await
    }

    /// Cleanup any previous operation.
    ///
    /// This must be called before trying to set a new operation, but
    /// is implicitly called by `completion` and `cancellation`.
    ///
    /// This should normally *not* be called directly but only implicitly
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

/// Type wrapping a `RABufferHandle` which has a currently ongoing operation.
pub struct OperationHandle<'a, V, OpInt>
where
    OpInt: OperationInteraction
{
    anchor: RABufferHandle<'a, V, OpInt>
}


impl<'a, 'r, V, OpInt> OperationHandle<'a, V, OpInt>
where
    OpInt: OperationInteraction
{

    /// See [`RABufferAnchor.completion()`]
    pub async fn completion(mut self) -> Option<OpInt::Result> {
        self.anchor.completion().await
    }

    /// See [`RABufferAnchor.cancellation()`]
    pub async fn cancellation(mut self) -> Option<OpInt::Result> {
        self.anchor.cancellation().await
    }

    /// See [`RABufferAnchor.request_cancellation()`]
    pub async fn request_cancellation(&mut self) {
        self.anchor.request_cancellation().await;
    }

    /// See [`RABufferAnchor.operation_interaction()`]
    pub fn operation_interaction(&self) -> Option<Pin<&OpInt>> {
        self.anchor.operation_interaction()
    }
}



#[macro_export]
macro_rules! ra_buffer_anchor {
    ($name:ident = [$init:literal; $len:literal] of $OpInt:ty) => (
        let mut $name = [$init; $len];
        // SAFE:
        // 1. We can use the buffer as it's directly on the stack above the anchor
        // 2. We directly pin the anchor to the stack as it's required.
        let mut $name = unsafe { RABufferAnchor::<_, $OpInt>::new_unchecked(&mut $name) };
        // SAFE:
        // 1. Works like `pin_mut!` we shadow the same stack allocated buffer to prevent any non-pinned
        //    access to it.
        //FIXME: Update SAFE doc
        let mut $name = unsafe { RABufferHandle::new_unchecked(&mut $name) };
    );
}


#[cfg(feature="embedded-dma")]
const _: () =  {
    use embedded_dma::{ReadBuffer, WriteBuffer, Word};

    unsafe impl<'r, 'a, V, OpInt> ReadBuffer for Pin<&'r mut RABufferAnchor<'a, V, OpInt>>
    where
        V: Word + 'a,
        OpInt: OperationInteraction + 'a,
    {
        type Word = V;
        unsafe fn read_buffer(&self) -> (*const V, usize) {
            let (ptr, len) = self.reborrow().get_buffer_ptr_and_len();
            (ptr as *const V, len)
        }
    }

    unsafe impl<'r, 'a, V, OpInt> WriteBuffer for Pin<&'r mut RABufferAnchor<'a, V, OpInt>>
    where
        V: Word + 'a,
        OpInt: OperationInteraction + 'a,
    {
        type Word = V;
        unsafe fn write_buffer(&mut self) -> (*mut V, usize) {
            self.reborrow().get_buffer_ptr_and_len()
        }
    }

};


#[cfg(test)]
mod mock_operation;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    mod usage_patterns {
        use core::mem;
        use super::super::*;
        use crate::mock_operation::*;

        #[async_std::test]
        async fn leaked_operations_get_canceled_and_ended_before_new_operations() {
            let mi = async {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);

                // If we leaked the op we still can poll on the buffer directly
                let mi = call_and_leak_op(buffer.reborrow()).await;
                mi.assert_not_run();
                buffer.reborrow().completion().await;
                mi.assert_completion_run();

                // If we create a new operation while one is still running the old one is first canceled.
                let old_mi = call_and_leak_op(buffer.reborrow()).await;
                old_mi.assert_not_run();
                let mi = call_and_leak_op(buffer.reborrow()).await;
                old_mi.assert_cancellation_run();
                mi.assert_not_run();
                buffer.reborrow().cancellation().await;
                mi.assert_cancellation_run();

                {
                    // Doing re-borrows with as_mut() can be useful
                    // to avoid lifetime/move problems.
                    let mut buffer = buffer.reborrow();
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


                let (mut op, mi) = call_op(buffer).await;
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
            }.await;
            // ...
            // So here after the drop the op has ended. (As a side note,
            // assert_cancellation/completion_run calls assert_end_op_check_run).
            mi.assert_op_ended_enforced();
        }

        async fn call_and_leak_op<'a>(buffer: RABufferHandle<'a, u8, OpIntMock>) -> MockInfo {
            let (op, mock_info) = mock_operation(buffer).await;
            mem::forget(op);
            mock_info
        }

        async fn call_op<'a>(buffer: RABufferHandle<'a, u8, OpIntMock>) -> (OperationHandle<'a, u8, OpIntMock>, MockInfo) {
            mock_operation(buffer).await
        }
    }

    mod RABufferAnchor {

        mod try_register_new_operation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn fails_if_a_operation_is_still_pending() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);

                let (_, mock) = mock_operation(buffer.reborrow()).await;

                let (ptr, len) = buffer.reborrow().get_buffer_ptr_and_len();
                let (op_int, _, new_mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.reborrow().try_register_new_operation(op_int) };
                assert!(res.is_err());
                mock.assert_not_run();
                new_mock.assert_not_run();
            }

            #[async_std::test]
            async fn sets_the_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.reborrow().get_buffer_ptr_and_len();
                let (op_int, _, mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.reborrow().try_register_new_operation(op_int) };
                assert!(res.is_ok());
                assert!(buffer.reborrow().has_pending_operation());
                mock.assert_not_run();
            }
        }

        mod cleanup_operation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn does_not_change_anything_if_there_was_no_pending_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.reborrow().get_buffer_ptr_and_len();
                let has_op = buffer.reborrow().has_pending_operation();

                buffer.reborrow().cleanup_operation();

                let (ptr2, len2) = buffer.reborrow().get_buffer_ptr_and_len();
                let has_op2 = buffer.reborrow().has_pending_operation();

                assert_eq!(ptr, ptr2);
                assert_eq!(len, len2);
                assert_eq!(has_op, has_op2);
            }

            #[async_std::test]
            async fn does_make_sure_the_operation_completed_without_moving() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_op, mock) = mock_operation(buffer.reborrow()).await;
                let op_int_addr = buffer
                    .operation_interaction()
                    .map(|pin| pin.get_ref() as *const _)
                    .unwrap();
                buffer.reborrow().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
            }

            #[async_std::test]
            async fn does_drop_the_operation_in_place() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_op, mock) = mock_operation(buffer.reborrow()).await;
                let op_int_addr = buffer.reborrow()
                    .operation_interaction()
                    .map(|pin| pin.get_ref() as *const _)
                    .unwrap();
                buffer.reborrow().cleanup_operation();
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
                let buff_ptr = buffer as *mut _ as  *mut u8;
                let buff_len = buffer.len();

                let mut anchor = unsafe { RABufferAnchor::<_, OpIntMock>::new_unchecked(buffer) };
                let anchor = unsafe { RABufferHandle::new_unchecked(&mut anchor) };

                let (ptr, len) = anchor.get_buffer_ptr_and_len();
                assert_eq!(len, buff_len);
                assert_eq!(ptr, buff_ptr);
            }
        }

        mod has_pending_operation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn returns_true_if_there_is_a_not_cleaned_up_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                assert!(not(buffer.reborrow().has_pending_operation()));
                let (op, _mock) = mock_operation(buffer.reborrow()).await;
                assert!(op.anchor.has_pending_operation());
                op.cancellation().await;
                assert!(not(buffer.reborrow().has_pending_operation()));
            }
        }

        mod request_cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn awaits_the_poll_request_cancellation_function_on_the_op_int_instance() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.request_cancellation().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod completion {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_completion_run();
            }

            #[async_std::test]
            async fn makes_sure_to_make_sure_operation_actually_did_end() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_completion_run();
                mock.assert_op_ended_enforced()
            }

            #[async_std::test]
            async fn makes_sure_to_clean_up_after_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_was_dropped();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation_and_complete() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.cancellation().await;
                mock.assert_cancellation_run();
            }


            #[async_std::test]
            async fn makes_sure_that_operation_ended() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                buffer.cancellation().await;
                mock.assert_cancellation_run();
                mock.assert_op_ended_enforced()
            }
            #[async_std::test]
            async fn makes_sure_to_clean_up() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                buffer.cancellation().await;
                mock.assert_was_dropped();
            }
        }

        mod buffer_mut {
            use super::super::super::*;
            use crate::mock_operation::*;

            #[async_std::test]
            async fn buffer_access_awaits_completion() {
                ra_buffer_anchor!(buffer = [12u32; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.reborrow()).await;
                let mut_ref = buffer.buffer_mut().await;
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
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (op, mock) = mock_operation(buffer.reborrow()).await;
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
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (mut op, mock) = mock_operation(buffer.reborrow()).await;
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
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (op, mock) = mock_operation(buffer.reborrow()).await;
                mock.assert_not_run();
                op.cancellation().await;
                mock.assert_cancellation_run();
            }
        }
    }
}

//TODO integration test with:
//  a slice of allocated memory
//  a future copied onto that memory
//  a thread based write to mem
// so that we can check if we can write to mem after drop