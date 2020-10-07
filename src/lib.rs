//! This library provides a way to have a stack allocated buffer which can be
//! use by completion based I/O like DMA, io-uring.
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
//! # Recommendation
//!
//! - `Pin<&mut RABufferAnchor<...>>` mostly works like a `&mut` but sadly
//!   doesn't automatically re-borrow. So e.g. calling `method(buffer); method(buffer);`
//!   doesn't work as `buffer` is moved into `method`. To get re-borrowing
//!   behavior like for a `&mut` you can use `.as_mut()`. E.g.
//!   `method(buffer.as_mut()); method(buffer.as_mut());`.
//!   Sadly that also means that for most methods on it you will use `.as_mut()`.
//!   E.g. `buffer.as_mut().completion().await`. Which is a bit annoying but
//!   currently a limitation of rust.
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
//! So e.g. [`RABufferAnchor.buffer_mut()`] hands out a `&mut [V]`
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
//! Furthermore it requires the anchor to only be accessible through the the `Pin`. (This
//! is best done the same way `pin_utils::pin_mut!` works.)
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
//! # use remote_accessed_buffer::{OperationInteraction, RABufferAnchor};
//! # struct DMAInteraction;
//! # unsafe impl OperationInteraction for DMAInteraction {
//! #   fn make_sure_operation_ended(self: Pin<&mut Self>) { todo!() }
//! #   fn poll_request_cancellation(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> { todo!() }
//! #   fn poll_completion(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> { todo!() }
//! # }
//! let mut buffer = [0u8; 32];
//! // SAFE:
//! // 1. We can use the buffer as it's directly on the stack above the anchor
//! // 2. We directly pin the anchor to the stack as it's required.
//! let mut buffer = unsafe { RABufferAnchor::<_, DMAInteraction>::new_unchecked(&mut buffer) };
//! // SAFE:
//! // 1. Works like `pin_mut!` we shadow the same stack allocated buffer to prevent any non-pinned
//! //    access to it.
//! let mut buffer = unsafe { Pin::new_unchecked(&mut buffer) };
//! ```
//!
//! Here by having the array buffer on the same stack as the anchor and making the buffer
//! out-life the anchor (it's on the stack "above" the buffer) we know that either we will
//! leak both the buffer and the anchor (which is ok) or the anchor will be dropped be the
//! buffer can maybe be accessed again (in this case it can not!).
//!
//! Semantically seen the anchor takes ownership of the buffer until it's dropped.
//!
//! Furthermore by shadowing the anchor with it's pin we make sure it's impossible to
//! access the anchor through anything but the `Pin`. If we would not have shadowed it
//! we would need to manually guarantee that it's not accessed which is just realy anoying
//! to do and error prone. Note that this aspects works exactly the same as `pin_mut!`.
//! I literally could have replaced the last line with `pin_mut!(buffer)` having one
//! unsafe line less in my code (but instead in `pin_mut!`). But for demonstration
//! purpose and readability writing it out by hand is better, here.
//!
//! Now one think which should be mentioned is that needing to wait for completion in
//! `Drop::drop` is not the best thing to do. We can not get around it but having something
//! like a `AsyncDropPreperation` trait which automatically does something like a async
//! drop before the drop would be supper helpful (a `AsyncDrop` which replaces `Drop` can't
//! really work as far as I know).
//!
//! Because of this it's strongly recommended to make sure that under normal circumstances
//! `buffer.completion().await` or `buffer.cancellation().await` is run before dropping it.
//!
//! ## How is it safe to drop the `OperationHandle`?
//!
//! If a operation started a `OperationInteraction` instance is registered on the buffer.
//! This instance can be used to check if the operation ended.
//!
//! This instance is *not* placed in the `OperationHandle` returned. Instead it is placed
//! "upstram" in the pinned anchor. This mean we can guarantee to have access to it until
//! we explicitly remove it because we realized the operation ended.
//!
//! The `OperationHandle` just wraps the used `Pin<&mut RABufferAnchor<...>>` to have a
//! way to encode that fact that there is a ongoing operation in the type system.
//!
//! You can always drop the `OperationHandle` handle which will literally have no affect
//! on the operation itself.
//!
//! The reason for this is that we anyway can access the `Pin<&mut RABufferAnchor<...>>`
//! while a operation is going on using re-borrows + (safe) leaking. In the end awaiting
//! completion/cancellation on the `OperationHandle` just forwards it the the same named
//! methods on `RABufferAnchor`!
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
//! There are two methods [`RABufferAnchor.buffer_mut()`] and [`RABufferAnchor.buffer_ref()`]
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
//!   a owner is enough for this. This can be e.g. a `Box<[V]>`,
//!   `Vec<V>` or even a `MutexGuard<[V]>` it's only important that it owns
//!   the buffer. We do not give out any drop guarantees for the buffer anyway,
//!   as we semantically "send" the buffer to the operation and receive it back
//!   once it's completed.
//!
//! - The buffer owner isn't required to be placed on the same stack frame. But it
//!   must be placed on the same or a parent stack frame and the same or a parent
//!   future. If and only if the future is pinned onto a normal stack the same or
//!   parent stack frame extends from inside the async stack to the outside as the
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
//!     let buffer = unsafe { RABufferAnchor::<_, DMAInteraction>::new_unchecked(&mut buffer); };
//!     pin_mut!(buffer)
//!     do_some_dma_magic(buffer.as_mut()).await;
//!     // make sure we properly await any pending operation aboves method might not have
//!     // awaited the completion on, so that we don't block on drop (through that kinda doesn't
//!     // matter in this example).
//!     buffer.completion().await
//! };
//! pin_mut!(future)
//! let result = block_on_pin(future);
//! ```
//!
//! While aboves example is fully fine, *it's strongly recommended to not do so*. As it's very
//! prone to introducing unsafe-contract breaches. E.g. just changing the last two lines to
//! `smol::block_on(future)` would brake the unsafe-contract as it no longer guarantees that
//! the buffer is on the same or an parent stack of the anchor.
//!
//!
//! ## Implementing Operations with this buffer
//!
//! This section contains some information for API implementers which do want to use this buffer.
//!
//! - The [`RABufferAnchor.try_register_new_operation()`] method can be used to register a new operation.
//!
//! - As it will fail if there is still a ongoing operation ist recommended that the method calling register
//!   does a `buffer.as_mut().cancellation().await`.  In some special cases a `buffer.as_mut().completion().await`
//!   can be appropriate to. But generally starting a new operation should cancel ongoing operations which have
//!   not yet been completed if possible. Only the [`RABufferAnchor.buffer_mut()`] and [`RABufferAnchor.buffer_ref()`]
//!   methods should to a implicit awaiting of completion (Which they do as there behavior is not generic.)
//!
//! - A operation must only start *after* [`RABufferAnchor.try_register_new_operation()`] returned successfully.
//!
//! - Semantically the ownership of the buffer is passed to whatever executes the operation until it completes.
//!   This is also why the anchor stores a pointer instead of a reference to the buffer.
//!
//! - To make it easier to build operations the [`RABufferAnchor.get_buffer_ptr_and_len()`] methods
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
//!   is called before dropping it. By combining this knowledge with a (unsafe) cell or some other type having
//!   internal mutability it's possible to pass pointers *into* the `OperationInteraction` instance to whatever
//!   does execute the operation. For mor details see below. Be *warned* that to do something like this you
//!   **always** need interior mutability as we do use `Pin<&mut OpInt>` references.
//!
//! - The [`RABufferAnchor.operation_interaction()`] method can be used to get a pinned borrow to the current
//!   operation interaction.
//!
//! - The operation interaction instance is *not* static but must outlive the anchor. (FIXME: And I think due to
//!   implementation details the buffer, too? At least for now.). This means something like a `&mut Channel`
//!   singleton instance describing a DMA channel should be able to be passed into the operation as part of
//!   the [`OperationInteraction`] instance.
//!
//! - Whatever is used to implement an operation should make sure that it *does not leak it's way to notify
//!   that the operation completed*. Because if it does we have the problem that we either will have a permanently
//!   pending future or a permanently hanging `drop` method. Which both are really really bad. (This is the
//!   price for temporary handing out ownership of stack allocated buffers).
//!
//! Below is the **pseudo-code** of how a function starting a DMA transfer might look:
//!
//! FIXME: Test that pseudo code in a integration test.
//!
//! ```ignore
//! // call like `start_dma_operation(buffer.as_mut(), Direction::FromMemory, Periphery::FooBarDataPort).await;`
//! // FIXME lifetimes could have better out-life relationships.
//! // FIXME without a way to move the channel&target back out this API sucks kinda ESPECIALLY with
//! //       the lifetime constraint but this will be solved soon.
//! async fn start_dma_operation<'r, 'a, T, C >(
//!     mut buffer: Pin<&'r mut RABufferAnchor<'a, T::Word, DMAInteraction>>,
//!     direction: Direction,
//!     channel: &'a mut C, //the lifetime sucks bad time, we will fix that in newer version
//!     target: &'a mut T
//! ) -> OperationHandle
//! where
//!     T: dma::Target + 'a,
//!     C: dma::Channel + 'a
//! {
//!     buffer.cancellation().await;
//!     let (ptr, len) = buffer.as_mut().get_buffer_ptr_and_len();
//!     let interaction = DMAInteraction::new(ptr, len, channel, target);
//!     // Safe: We register the right interaction.
//!     let operation_handle = unsafe { buffer.as_mut().try_register_new_operation(interaction) };
//!     // SAFE(unwrap): We made sure the operation ended by polling cancellation
//!     let mut operation_handle = operation_handle.unwrap();
//!     let op_int = operation_handle.as_mut().operation_interaction()
//!     // SAFE: We must only call `start` once before it started.
//!     unsafe { op_int.start() };
//!     operation_handle
//! }
//! ```
//!
//! With `DMAInteraction` being something along the line of following **pseudo-code**:
//!
//! ```ignore
//! //FIXME C & T should *not* be part of the interface, sure a `dyn C`/`dyn T` would be
//! //      ok but we want to be able to use the same buffer for different channels and
//! //      target combinations, so making the `OperationInteraction` generic over the
//! //      channel and target is a very bad idea.
//! struct DMAInteraction<'a, C, T>
//! where
//!     // The Unpin is NOT NECESSARY it just makes the pseudo-code simpler
//!     C: dma::Channel + Unpin + 'a,
//!     T: dma::Target + Unpin + 'a
//! {
//!     ptr: *mut T::Word,
//!     len: usize,
//!     channel: &'a mut C,
//!     target: &'a mut T,
//! }
//!
//! impl<'a, C, T> DMAInteraction<'a, C, T>
//! where
//!     C: dma::Channel + Unpin + 'a,
//!     T: dma::Target + Unpin + 'a,
//! {
//!     fn new(ptr: *mut T::Word, len: usize, channel: &'a mut C, target: &'a mut T) -> Self {
//!         Self { ptr, len, channel, target }
//!     }
//!
//!     unsafe fn start(self: Pin<&mut Self>) {
//!         let self_ = self.get_mut();
//!         self_.target.enable_dma();
//!         let channel = &mut self._channel;
//!         channel.reset();
//!         channel.set_periphery(&mut *self_.target);
//!         channel.set_memory_address(as_address(ptr));
//!         channel.set_transfer_length(len);
//!         channel.set_word_size(T::Word::size());
//!         channel.enable();
//!     }
//!
//!     fn is_completed(&mut self) -> bool {
//!         self_.event_occurred(Event::Completed) || self_.event_ocurred(Event::Failed)
//!     }
//! }
//!
//! impl<'a, C, T> OperationInteraction for DMAInteraction<'a, C, T>
//! where
//!     C: dma::Channel + Unpin + 'a,
//!     T: dma::Target + Unpin + 'a,
//! {
//!     fn make_sure_operation_ended(self: Pin<&mut Self>) {
//!         let self_ = self.get_mut(); // or p
//!         while !self_.is_completed() {
//!             spin_loop_hint()
//!             //or thread::yield()
//!         }
//!
//!         //TODO we could "clear" events here or reset the channel.
//!         //TODO instead of `&'a mut C/T` we could pass them in and
//!         //     "magically" move them out at this palace.
//!     }
//!
//!     fn poll_request_cancellation(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<()> {
//!         let self_ = self.get_mut();
//!         if self_.channel.is_enabled() {
//!             self_.channel.disable();
//!         }
//!     }
//!
//!     // There are better ways to do this (using interrupts + wakers) but for this
//!     // example this is enough.
//!     fn poll_completion(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> {
//!         let self_ = self.get_mut();
//!         if self_.is_completed() {
//!             Poll::Ready(())
//!         } else {
//!             // We don't wake from external source (e.g. interrupt) so we need to
//!             // waker now so that it doest become a "zombie" task
//!             cx.waker().wake_by_ref();
//!             Poll::Pending;
//!         }
//!     }
//! }
//! ```
//!
//! ### Handling task waking and completion awaiting.
//!
//! In the aboves pseudo-code we busy poll the for the completion of an DMA-Operation,
//! which is not very good and we can do better.
//!
//! Like already mentioned the `OperationInteraction` instance is itself pinned.
//! This means by using interior mutability on some field of it we can provide
//! some memory shared between whatever executes the task and our-side which polls
//! on the task. Naturally we need to use some for of synchronization. But a `atomic`
//! is often good enough.
//!
//! With that we can do something on *similar* to:
//!
//! - every time `poll_request_cancellation` or `poll_completion` is called we do:
//!   1. Check if the action already completed, if so return `Poll::Ready`
//!   2. If not clone the waker (`cx.waker().clone()`) and replace it as the
//!      new current waker (guarded with some atomic based spin lock or similar)
//!      - We might be able to optimize this by only cloning the new waker if it has
//!        changed.
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
//! This area need a bit more prototyping.
//!
//!
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

mod utils;

use core::{marker::PhantomData, mem::ManuallyDrop, pin::Pin, ptr, task::Context, task::Poll};
use crate::utils::abort_on_panic;
/// Trait for type allowing interaction with an ongoing operation
///
/// # Unsafe-Contract
///
/// The implementor MUST guarantee that after `make_sure_operation_ended`
/// the operation did end and the buffer is no longer accessed in ANY way
/// by the operation.
///
/// See the method documentation of `make_sure_operation_ended` for more
/// details.
pub unsafe trait OperationInteraction {

    /// This method only returns when the operation ended.
    ///
    /// This method is always called before cleaning up an operation.
    ///
    /// It is guaranteed to be called after [`OperationInteraction.poll_completion()`] returns `Ready`.
    ///
    /// It is guaranteed to be called before this [`OperationInteraction`] instance is dropped, if it's
    /// dropped from inside of the anchor.
    ///
    /// It is also always called when the anchor is dropped and there is a operation which has
    /// not yet been cleaned up.
    ///
    /// If possible this method should try to cancel any ongoing operation so that it can
    /// return as fast as possible.
    ///
    /// This method will be called both in sync and async code. It MUST NOT depend on environmental
    /// context provided by an async runtime/executor/scheduler or similar.
    ///
    /// It MUST be no problem to call this method more then once.
    ///
    /// Due to lifetimes and mutule borrows this method will never be called more then once
    /// concurrently, only more then once sequentially.
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
    /// Note that besides on `Drop` of the anchor `poll_complete` will always
    /// be polled before so if you can no longer say if it completed or not
    /// it's can be better to hang the future and with that permanently leak
    /// the buffer instead of hanging the thread.
    fn make_sure_operation_ended(self: Pin<&mut Self>);

    /// Notifies the operation that it should be canceled.
    ///
    /// Once a future using this as poll completes it means that
    /// the request for cancellation has been received be the
    /// operation. It *does not* mean has ended through cancellation.
    ///
    /// As many operations do not support cancellation this has a
    /// default implementation which instantly completes.
    ///
    /// # Extended poll guarantees.
    ///
    /// A implementor must guarantee that polling this *even after
    /// poll returned `Ready`* doesn't panic. Wrt. this this differs
    /// from a classical poll.
    ///
    fn poll_request_cancellation(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<()> {
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
    fn poll_completion(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()>;

}

//Note: I could use #[pin_project] but I have additional
//      restrictions for `operation_interaction` and need
//      to project into the option. So it's not worth it.
pub struct RABufferAnchor<'a, V, OpInt>
where
    V: 'a,
    OpInt: OperationInteraction + 'a
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
    /// # Pin Safety (wrt. Pin<&mut Self>)
    ///
    /// This type guarantees that there are no inner references or similar
    /// to this field.
    ///
    /// If it is `None` it always can safely be replaced with `Some`.
    ///
    /// This field should never be `Some` if `Self` is not pinned.
    ///
    /// If it's `Some` we MUST treat it roughly as if it's pinned.
    ///
    /// So if we want to set it to none or replace it with another
    /// operation we MUST do following:
    ///
    /// 1. first calling `make_sure_operation_ended` on it **without** moving it
    /// 2. dropping it in place in `ManuallyDrop::drop`
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
    operation_interaction: Option<ManuallyDrop<OpInt>>
}

impl<'a, V, OpInt> RABufferAnchor<'a, V, OpInt>
where
    V: 'a,
    OpInt: OperationInteraction  + 'a
{

    /// Create a new instance with given buffer.
    ///
    /// # Unsafe-Contract
    ///
    /// 1. You can pass any buffer in where you guarantee that it only can be re-purposed
    ///   after the anchor is dropped and that if the anchor is leaked the buffer is
    ///   guaranteed to be leaked, too.
    ///
    /// 2. You must `Pin` the anchor and make sure it's pinned in a way that `1.` is
    ///    still uphold.
    ///
    /// It **very strongly** recommended always do following:
    ///
    /// 1. Have the buffer on the stack directly above the anchor.
    /// 2. `Pin` the anchor to the stack immediately after constructing it
    /// 3. Use the `Pin` to shadow the anchor.
    ///
    /// This is the most simple way to guarantee the unsafe contract is uphold.
    ///
    /// See module level documentation for more details.
    pub unsafe fn new_unchecked(buffer: &'a mut [V]) -> Self {
        RABufferAnchor {
            buffer_type_hint: PhantomData,
            buffer: (buffer as *mut _ as *mut V, buffer.len()),
            operation_interaction: None
        }
    }


    /// Returns a mut reference to the underling buffer.
    ///
    /// If a operations is currently in process it first awaits the end of the operation.
    ///
    /// This will not try to cancel any ongoing operation. If you need that you
    /// should await [`RABuffer::request_cancellation()`] before caling this
    /// method.
    pub async fn buffer_mut(mut self: Pin<&mut Self>) -> &mut [V] {
        self.as_mut().completion().await;
        // Safe: We have a (pinned) &mut borrow to the anchor and we made
        //       sure it's completed (completion always calls `cleanup_operation`).
        unsafe {
            let (ptr, len) = self.get_unchecked_mut().buffer;
            core::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Returns a reference to the underling buffer.
    ///
    /// If a operations is currently in process it first awaits the end of the operation.
    ///
    /// This will not try to cancel any ongoing operation. If you need that you
    /// should await [`RABuffer::request_cancellation()`] before caling this
    /// method.
    pub async fn buffer_ref(mut self: Pin<&mut Self>) -> &[V] {
        self.as_mut().completion().await;
        // Safe: We have a (pinned) &mut borrow to the anchor and we made
        //       sure it's completed (completion always calls `cleanup_operation`).
        unsafe {
            let (ptr, len) = self.get_unchecked_mut().buffer;
            core::slice::from_raw_parts(ptr, len)
        }
    }


    /// Return a pointer to the start of the the underlying buffer and it's size.
    ///
    ///
    ///
    /// # Safety
    ///
    /// You must guarantee that you only use the pointer after you installed
    /// a appropriate `OperationInteraction` with [`RABufferAnchor.try_register_new_operation()`].
    ///
    /// The reason why we have this unsafe method instead of returning the pointer
    /// from [`RABufferAnchor.try_register_new_operation()`] is because you
    /// might need it to create the `OperationInteraction` instance.
    pub fn get_buffer_ptr_and_len(self: Pin<&mut Self>) -> (*mut V, usize) {
        // Safe: It's unsafe to access the pointer but safe to create it.
        unsafe { self.get_unchecked_mut().buffer }
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
    pub unsafe fn try_register_new_operation<'r>(mut self: Pin<&'r mut Self>, new_opt_int: OpInt) -> Result<OperationHandle<'a, 'r, V, OpInt>, ()> {
        if self.as_ref().has_pending_operation() {
            Err(())
        } else {
            // Safe: We know it's `None`, a `None` value on `operation_interaction`
            //       has explicitly no pinning guarantees in this types unsafe
            //       contract. As such we can "just" set it to `Some(..)`
            {
                let opt_pin = &mut self.as_mut().get_unchecked_mut().operation_interaction;
                debug_assert!(opt_pin.is_none());
                *opt_pin = Some(ManuallyDrop::new(new_opt_int));
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
    pub fn has_pending_operation(self: Pin<&Self>) -> bool {
        self.get_ref().operation_interaction.is_some()
    }

    /// Projection allowing access to the internally stored [`OperationInteraction`] instance.
    ///
    /// Returns `None` if there is no ongoing interaction.
    pub fn operation_interaction(self: Pin<&mut Self>) -> Option<Pin<&mut OpInt>> {
        // SAFE: Due to the special guarantees we give to the operation_interaction
        //       field it is safe to create a `Option<Pin<&mut OpInt>>` from it.
        //       Through we need to give some guarantees around it like we must not
        //       move out the `OpInt` in the drop implementation (which is why it's
        //       wrapped in ManuallyDrop so that we can drop it in place).
        unsafe  {
            self.get_unchecked_mut()
                .operation_interaction
                .as_mut()
                .map(|op_int| Pin::new_unchecked(&mut **op_int))
        }
    }

    /// If there is an ongoing operation notify it to be canceld.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.
    pub async fn request_cancellation(self: Pin<&mut Self>) {
        if let Some(mut op_int) = self.operation_interaction() {
            futures_lite::future::poll_fn(|ctx| op_int.as_mut().poll_request_cancellation(ctx)).await;
        }
    }

    /// If there is an ongoing operation await the completion of it.
    ///
    /// This will await `op_int.poll_completion` and then call `cleanup_operation`.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.ge).
    pub async fn completion(mut self: Pin<&mut Self>) {
        if let Some(mut op_int) = self.as_mut().operation_interaction() {
            futures_lite::future::poll_fn(|ctx| op_int.as_mut().poll_completion(ctx)).await;
        }
        //WARNING: Some other internal methods might rely on completion always calling
        //         `self.cleanup_operation()` at the end! So don't remove it it might
        //         brake the unsafe-contract.
        self.cleanup_operation();
    }

    /// If there is an ongoing operation notify it to be canceled and await the completion of it.
    ///
    /// If it can not be canceled this will just wait normal completion.
    ///
    /// This will first await `op_int.poll_request_cancellation`, then `op_int.poll_completion`
    /// and then calls `cleanup_operation`.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.
    pub async fn cancellation(mut self: Pin<&mut Self>) {
        self.as_mut().request_cancellation().await;
        self.completion().await;
    }

    /// Cleanup any previous operation.
    ///
    /// This must be called before trying to set a new operation, but
    /// is implicitly called by `completion` and `cancellation`.
    ///
    /// This should normally *not* be called directly but only implicitly
    /// through `completion().await` or `cancellation().await`. Through there
    /// are some supper rare edge cases where exposing it is use-full.
    pub fn cleanup_operation(mut self: Pin<&mut Self>) {
        if let Some(mut opt_int) = self.as_mut().operation_interaction() {
            abort_on_panic(|| opt_int.as_mut().make_sure_operation_ended());
            // Safe:
            //   1. We called `make_sure_operation_ended`.
            //   2. We drop it in-place without moving it.
            //   3. We will override the now "cobbled" option field
            //      with `None` which is fine as we already dropped the
            //      pinned value and pin must only hold until it's dropped.
            unsafe {
                ptr::drop_in_place(opt_int.get_unchecked_mut());
            }
        }
        // Safe: Either it's already `None` or we just dropped the inner (ManuallyDrop wrapped) value and
        //       as such ended it's pinning constraint.
        unsafe { self.get_unchecked_mut().operation_interaction = None; }
    }

}

impl<'a, V, OpInt> Drop for RABufferAnchor<'a, V, OpInt>
where
    V: 'a,
    OpInt: OperationInteraction + 'a
{
    fn drop(&mut self) {
        // Safe: We are about to drop self and can guarantee it's no longer moved before drop
        let pinned = unsafe { Pin::new_unchecked(self) };
        pinned.cleanup_operation();
    }
}

/// Type wrapping a `Pin<&mut RABufferAnchor<...>>` which has a currently ongoing operation.
pub struct OperationHandle<'a, 'r, V, OpInt>
where
    V: 'a,
    OpInt: OperationInteraction + 'a
{
    anchor: Pin<&'r mut RABufferAnchor<'a, V, OpInt>>
}


impl<'a, 'r, V, OpInt> OperationHandle<'a, 'r, V, OpInt>
where
    V: 'a,
    OpInt: OperationInteraction + 'a
{

    /// See [`RABufferAnchor.completion()`], through this gives out a `Pin<&mut RABufferAnchor<...>>`
    pub async fn completion(mut self) -> Pin<&'r mut RABufferAnchor<'a, V, OpInt>> {
        self.anchor.as_mut().completion().await;
        self.anchor
    }

    /// See [`RABufferAnchor.cancellation()`], through this gives out a `Pin<&mut RABufferAnchor<...>>`
    pub async fn cancellation(mut self) -> Pin<&'r mut RABufferAnchor<'a, V, OpInt>> {
        self.anchor.as_mut().cancellation().await;
        self.anchor
    }

    /// See [`RABufferAnchor.request_cancellation()`]
    pub async fn request_cancellation(&mut self) {
        self.anchor.as_mut().request_cancellation().await;
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
        let mut $name = unsafe { Pin::new_unchecked(&mut $name) };
    );
}


#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    #[path="./mock_operations.rs"]
    mod mock_operation;

    mod usage_patterns {
        use core::mem;
        use super::super::*;
        use super::mock_operation::*;

        #[async_std::test]
        async fn leaked_operations_get_canceled_and_ended_before_new_operations() {
            let mi = async {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);

                // If we leaked the op we still can poll on the buffer directly
                let mi = call_and_leak_op(buffer.as_mut()).await;
                mi.assert_not_run();
                buffer.as_mut().completion().await;
                mi.assert_completion_run();

                // If we create a new operation while one is still running the old one is first canceled.
                let old_mi = call_and_leak_op(buffer.as_mut()).await;
                old_mi.assert_not_run();
                let mi = call_and_leak_op(buffer.as_mut()).await;
                old_mi.assert_cancellation_run();
                mi.assert_not_run();
                buffer.as_mut().cancellation().await;
                mi.assert_cancellation_run();

                {
                    // Doing re-borrows with as_mut() can be useful
                    // to avoid lifetime/move problems.
                    let  buffer = buffer.as_mut();
                    let (op, mi) = call_op(buffer).await;
                    mi.assert_not_run();
                    // A buffer mut ref we can reuse.
                    // Note that it has exact the same lifetime as
                    // the moved buffer mut ref 3 lines above.
                    let buffer = op.completion().await;
                    mi.assert_completion_run();

                    let (op, mi) = call_op(buffer).await;
                    mi.assert_not_run();
                    let buffer = op.cancellation().await;
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

        async fn call_and_leak_op<'r, 'a>(buffer: MockBufferMut<'r,'a>) -> MockInfo {
            let (op, mock_info) = mock_operation(buffer).await;
            mem::forget(op);
            mock_info
        }

        async fn call_op<'r, 'a>(buffer: MockBufferMut<'r, 'a>) -> (OperationHandle<'a,'r, u8, OpIntMock>, MockInfo) {
            mock_operation(buffer).await
        }
    }

    mod RABufferAnchor {

        mod try_register_new_operation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn fails_if_a_operation_is_still_pending() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);

                let (_, mock) = mock_operation(buffer.as_mut()).await;

                let (ptr, len) = buffer.as_mut().get_buffer_ptr_and_len();
                let (op_int, _, new_mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.as_mut().try_register_new_operation(op_int) };
                assert!(res.is_err());
                mock.assert_not_run();
                new_mock.assert_not_run();
            }

            #[async_std::test]
            async fn sets_the_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.as_mut().get_buffer_ptr_and_len();
                let (op_int, _, mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.as_mut().try_register_new_operation(op_int) };
                assert!(res.is_ok());
                assert!(buffer.as_ref().has_pending_operation());
                mock.assert_not_run();
            }
        }

        mod cleanup_operation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn does_not_change_anything_if_there_was_no_pending_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.as_mut().get_buffer_ptr_and_len();
                let has_op = buffer.as_ref().has_pending_operation();

                buffer.as_mut().cleanup_operation();

                let (ptr2, len2) = buffer.as_mut().get_buffer_ptr_and_len();
                let has_op2 = buffer.as_ref().has_pending_operation();

                assert_eq!(ptr, ptr2);
                assert_eq!(len, len2);
                assert_eq!(has_op, has_op2);
            }

            #[async_std::test]
            async fn does_make_sure_the_operation_completed_without_moving() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_op, mock) = mock_operation(buffer.as_mut()).await;
                let op_int_addr = buffer.as_mut()
                    .operation_interaction()
                    .map(|pin| pin.get_mut() as *mut _)
                    .unwrap();
                buffer.as_mut().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
            }

            #[async_std::test]
            async fn does_drop_the_operation_in_place() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_op, mock) = mock_operation(buffer.as_mut()).await;
                let op_int_addr = buffer.as_mut()
                    .operation_interaction()
                    .map(|pin| pin.get_mut() as *mut _)
                    .unwrap();
                buffer.as_mut().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
                mock.assert_was_dropped();
            }
        }

        mod get_buffer_ptr_and_len {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn return_the_right_pointer_and_len() {
                let mut buffer = [0u8; 32];
                let buffer = &mut buffer;
                let buff_ptr = buffer as *mut _ as  *mut u8;
                let buff_len = buffer.len();

                let mut anchor = unsafe { RABufferAnchor::<_, OpIntMock>::new_unchecked(buffer) };
                let anchor = unsafe { Pin::new_unchecked(&mut anchor) };

                let (ptr, len) = anchor.get_buffer_ptr_and_len();
                assert_eq!(len, buff_len);
                assert_eq!(ptr, buff_ptr);
            }
        }

        mod has_pending_operation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn returns_true_if_there_is_a_not_cleaned_up_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                assert!(not(buffer.as_ref().has_pending_operation()));
                let (op, _mock) = mock_operation(buffer.as_mut()).await;
                assert!(op.anchor.as_ref().has_pending_operation());
                op.cancellation().await;
                assert!(not(buffer.as_ref().has_pending_operation()));
            }
        }

        mod request_cancellation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn awaits_the_poll_request_cancellation_function_on_the_op_int_instance() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.request_cancellation().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod completion {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_completion_run();
            }

            #[async_std::test]
            async fn makes_sure_to_make_sure_operation_actually_did_end() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_completion_run();
                mock.assert_op_ended_enforced()
            }

            #[async_std::test]
            async fn makes_sure_to_clean_up_after_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.completion().await;
                mock.assert_was_dropped();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation_and_complete() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.cancellation().await;
                mock.assert_cancellation_run();
            }


            #[async_std::test]
            async fn makes_sure_that_operation_ended() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.cancellation().await;
                mock.assert_cancellation_run();
                mock.assert_op_ended_enforced()
            }
            #[async_std::test]
            async fn makes_sure_to_clean_up() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                buffer.cancellation().await;
                mock.assert_was_dropped();
            }
        }

        mod buffer_mut {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn buffer_access_awaits_completion() {
                ra_buffer_anchor!(buffer = [12u32; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                let mut_ref = buffer.buffer_mut().await;
                assert_eq!(mut_ref, &mut [12u32; 32] as &mut [u32]);
                mock.assert_completion_run();
                mock.assert_was_dropped();
            }
        }

        mod buffer_ref {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn buffer_access_awaits_completion() {
                ra_buffer_anchor!(buffer = [12u32; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                let a_ref = buffer.buffer_ref().await;
                assert_eq!(a_ref, &[12u32; 32] as &[u32]);
                mock.assert_completion_run();
                mock.assert_was_dropped();
            }
        }

    }

    mod OperationHandle {

        mod completion {
            use super::super::super::*;
            use super::super::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (op, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                op.completion().await;
                mock.assert_completion_run();
            }
        }

        mod request_cancellation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (mut op, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                op.request_cancellation().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_request_cancellation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (op, mock) = mock_operation(buffer.as_mut()).await;
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