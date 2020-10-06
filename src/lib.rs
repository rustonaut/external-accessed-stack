//!
//! # Design Idea
//!
//! 1. Pin can gives us a drop guarantee that the pinned memory is not re-purposed without
//!    calling drop in the pinned type.
//! 2. For anything on the stack this implies`?Leak` as the stack shrinking/growing repurposed
//!    memory.
//! 3. But we don't need `?Leak` on just he buffer but also a type which owns/control the buffer
//!    and manage the DMA (or similar).
//! 4. So something like `struct DMABuffer { dma_manager: DMAManager, dma_buffer: [u8] }` would
//!    be what we would want to pin. Problem is that we can't but variable sized struct's on the
//!    stack. Even with the unstable nightly feature we can not create a variable sized struct on
//!    the stack only variable length arrays and moving other variable sized types onto the stack
//!    (but we can work around it by having a [u8] VLA which we retype + to DMABuffer)
//! 5. Alternatively we could have a `struct DMABuffer<const LEN: usize>` but not only means this
//!    const buffer length but also const generics are as far as I can tell far from stable or
//!    properly working.
//! 6. So the new idea is we create a anchor which has a reference to a buffer and we will pin
//!    that anchor and through the unsafe constructor of the anchor we will extend the pinning
//!    to the buffer.
//!    - As a safe abstraction we will have a macro which makes sure to stack allocate the buffer in
//!      the same stack scope as the anchor and shadow it. With that we should be able to treat
//!      the buffer and other data as if pinned together.
//!    - Due to the Pin on stack guarantees we know the destructor will run before the stack is freed and
//!      the stack memory is potentially repurposed.
//!    - With that we can use the `RABufferAnchor`'s `Drop` implementation to make sure to halt destruction
//!      until DMA stopped and as such prevent stack clobbering.
//!    - We can have 2 APIs one which consumes the RABufferHandle and gives it back once the operation completes
//!      but in that case it can be a bit annoying to have to propagate the handles back up. The other is
//!      borrow based but in this case leaking the operation allows restarting a new operation before the
//!      old operation completes so magic is needed to handle that. (move a way to cancle into the pinned
//!      anchor)
//!
//!
//!
//! ### Handling the Operation Feedback
//!
//! For the borrow based API we need to store data in the anchor which can help us to await
//! if the operation has ended, to trigger the operation to cancel and to have a blocking
//! way to wait for the end of the operation and potentially cancel it on the way.
//!
//! For the `std` use-case this isn't to bad as we can have a `Box<dyn Trait>`.
//!
//! For the `non-std` use-case this is a bit more tricky as we don't want to have a `Box`.
//! So instead we need a well-know access method specific type *and* make the buffer specific
//! to the access method used. Depending on what we are working with awaiting completion
//! can also be very async-runtime specific as it might need a "await from interrupt" or
//! similar.
//!
//! So we need following instead:
//!
//! `ra_buffer_anchor(buffer1 = [0u8; 1024] for OperationInteraction);`
//!

use std::{marker::PhantomData, mem, pin::Pin, ptr, task::Context, task::Poll};

use mem::ManuallyDrop;

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

    ///
    /// - blocks until operation ended
    /// - should try to cancel if the operation did not yet ended and
    ///   the operation does support cancellation.
    /// - can be called after operation ended
    /// - must be callable in both sync anc async code (and yes it still blocks
    ///   in async code).
    /// - should (will?) not be called twice
    ///
    /// # Safety
    ///
    /// A implementor of this method must assure on a rust safety level that
    /// once this method returns the operation ended and there is no longer
    /// ANY access from it into the buffer. **Be aware that this includes
    /// return by panic.**
    ///
    /// As such if you for whatever reason can no longer be sure that this
    /// is the case you can only either hand the thread or abort the program.
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
    fn poll_cancel(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<()> {
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
    /// TODO explain why it works and what is needed
    ///
    /// - In practice: you *must* `Pin` the returned instance to the stack
    /// - This the returned instance is pinned this pin extends to the buffer passed in.
    ///   (WAIT WE DON'T need to extend the Pin to the buffer!!)
    /// - WAIT the buffer doesn't need to be on the stack or the same stack scope
    ///   in all cases (it matters for the fully leak boxed future case with a non 'static
    ///   box where the ref comes from the outside).
    /// - WAIT if we guarantee the buffer to be on the stack just above we don't need to
    ///   carry it's lifetime around.
    /// TODO TODO TODO TODO (require pin on stack etc.)
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
    /// should await [`RABuffer::notify_cancellation_intend()`] before caling this
    /// method.
    pub async fn buffer_mut(mut self: Pin<&mut Self>) -> &mut [V] {
        self.as_mut().completion().await;
        // Safe: We have a (pinned) &mut borrow to the anchor and we made
        //       sure it's completed (completion always calls `cleanup_operation`).
        unsafe {
            let (ptr, len) = self.get_unchecked_mut().buffer;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Returns a reference to the underling buffer.
    ///
    /// If a operations is currently in process it first awaits the end of the operation.
    ///
    /// This will not try to cancel any ongoing operation. If you need that you
    /// should await [`RABuffer::notify_cancellation_intend()`] before caling this
    /// method.
    pub async fn buffer_ref(mut self: Pin<&mut Self>) -> &[V] {
        self.as_mut().completion().await;
        // Safe: We have a (pinned) &mut borrow to the anchor and we made
        //       sure it's completed (completion always calls `cleanup_operation`).
        unsafe {
            let (ptr, len) = self.get_unchecked_mut().buffer;
            std::slice::from_raw_parts(ptr, len)
        }
    }


    /// Return a pointer to the start of the the underlying buffer and it's size.
    ///
    ///
    ///
    /// # Safety
    ///
    /// You must guarantee that you only use the pointer after you installed
    /// a appropriate `OperationInteraction` with [`RABufferAnchor.try_set_new_operation_interaction`].
    ///
    /// The reason why we have this unsafe method instead of returning the pointer
    /// from [`RABufferAnchor.try_set_new_operation_interaction`] is because you
    /// might need it to create the `OperationInteraction` instance.
    pub fn get_slice_start_ptr_and_len_unchecked(self: Pin<&mut Self>) -> (*mut V, usize) {
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
    pub unsafe fn try_set_new_operation_interaction<'r>(mut self: Pin<&'r mut Self>, new_opt_int: OpInt) -> Result<OperationHandle<'a, 'r, V, OpInt>, ()> {
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

    /// Returns a pint into the `Option::Some` of the  `operation_interaction` field.
    fn get_pin_to_op_int(self: Pin<&mut Self>) -> Option<Pin<&mut OpInt>> {
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
    pub async fn notify_cancellation_intend(self: Pin<&mut Self>) {
        if let Some(mut op_int) = self.get_pin_to_op_int() {
            futures_lite::future::poll_fn(|ctx| op_int.as_mut().poll_cancel(ctx)).await;
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
        if let Some(mut op_int) = self.as_mut().get_pin_to_op_int() {
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
    /// This will first await `op_int.poll_cancel`, then `op_int.poll_completion`
    /// and then calls `cleanup_operation`.
    ///
    /// This is normally called implicitly through the `OperationHandle`
    /// which is often wrapped in some operation specific type.
    ///
    /// Calling this directly is only possible if the `OperationHandle`
    /// has been leaked or detached.
    pub async fn cancellation(mut self: Pin<&mut Self>) {
        self.as_mut().notify_cancellation_intend().await;
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
        if let Some(mut opt_int) = self.as_mut().get_pin_to_op_int() {
            opt_int.as_mut().make_sure_operation_ended();
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

    /// Detach this `OperationHandle`.
    ///
    /// You still need to await completion or cancellation of this operation
    /// before starting any new operation.
    ///
    /// It fully safe and semantically acceptable to drop or leak the returned `Pin`.
    ///
    /// # Warning
    ///
    /// As you often need to do re-borrows it's easy for the returned `Pin` to have
    /// a smaller lifetime as the original `Pin`. This can lead to reduced UX when
    /// re-using the pin returned from detach. It's often (but not always) better to
    ///
    pub fn detach(self)  -> Pin<&'r mut RABufferAnchor<'a, V, OpInt>> {
        // Note: Trick to make sure we don't forget to update this if Self changes.
        let Self { anchor } = self;
        anchor
    }

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

    /// See [`RABufferAnchor.notify_cancellation_intend()`]
    pub async fn notify_cancellation_intend(&mut self) {
        self.anchor.as_mut().notify_cancellation_intend().await;
    }
}



#[macro_export]
macro_rules! ra_buffer_anchor {
    ($name:ident = [0u8; $len:literal] of $OpInt:ty) => (
        let mut $name = [0u8; $len];
        let mut $name = unsafe { RABufferAnchor::<_, $OpInt>::new_unchecked(&mut $name) };
        // Yirks lifetime folding prevents drop
        let mut $name = unsafe { Pin::new_unchecked(&mut $name) };
    );
}


#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    #[path="./mock_operations.rs"]
    mod mock_operation;

    mod usage_patterns {
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
                op.notify_cancellation_intend().await;
                mi.assert_notify_cancel_run();

                // Now here it gets interesting:
                //  We still have a ongoing operation which might already have stopped,
                //  but which isn't cleaned up (removing the op.notify_cancellation_intend above makes
                //  no difference `op.notify_cancellation_intend` might be a noop).
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

        mod try_set_new_operation_interaction {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn fails_if_a_operation_is_still_pending() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);

                let (_, mock) = mock_operation(buffer.as_mut()).await;

                let (ptr, len) = buffer.as_mut().get_slice_start_ptr_and_len_unchecked();
                let (op_int, _, new_mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.as_mut().try_set_new_operation_interaction(op_int) };
                assert!(res.is_err());
                mock.assert_not_run();
                new_mock.assert_not_run();
            }

            #[async_std::test]
            async fn sets_the_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.as_mut().get_slice_start_ptr_and_len_unchecked();
                let (op_int, _, mock) = OpIntMock::new(ptr, len);
                let res = unsafe { buffer.as_mut().try_set_new_operation_interaction(op_int) };
                assert!(res.is_ok());
                assert!(buffer.as_ref().has_pending_operation());
                mock.assert_not_run();
            }

            #[async_std::test]
            async fn returns_a_operation_handle_with_the_exact_same_lifetime_as_the_input_pin() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.as_mut().get_slice_start_ptr_and_len_unchecked();
                let (op_int, _, _mock) = OpIntMock::new(ptr, len);

                let mut slot = Some(buffer);
                let res = unsafe { slot.take().unwrap().try_set_new_operation_interaction(op_int) };
                let op = res.unwrap();
                // needs compatible lifetimes or it won't work
                slot = Some(op.detach());
                drop(slot)
            }
        }

        mod cleanup_operation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn does_not_change_anything_if_there_was_no_pending_operation() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (ptr, len) = buffer.as_mut().get_slice_start_ptr_and_len_unchecked();
                let has_op = buffer.as_ref().has_pending_operation();

                buffer.as_mut().cleanup_operation();

                let (ptr2, len2) = buffer.as_mut().get_slice_start_ptr_and_len_unchecked();
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
                    .get_pin_to_op_int()
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
                    .get_pin_to_op_int()
                    .map(|pin| pin.get_mut() as *mut _)
                    .unwrap();
                buffer.as_mut().cleanup_operation();
                mock.assert_op_int_addr_eq(op_int_addr);
                mock.assert_was_dropped();
            }
        }

        mod get_slice_start_ptr_and_len_unchecked {
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

                let (ptr, len) = anchor.get_slice_start_ptr_and_len_unchecked();
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

        mod notify_cancellation_intend {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn awaits_the_poll_cancel_function_on_the_op_int_instance() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                buffer.notify_cancellation_intend().await;
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
            async fn polls_op_int_poll_cancel_and_complete() {
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
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                let mut_ref = buffer.buffer_mut().await;
                assert_eq!(mut_ref, &mut [0u8; 32] as &mut [u8]);
                mock.assert_completion_run();
                mock.assert_was_dropped();
            }
        }

        mod buffer_ref {
            use super::super::super::*;
            use super::super::mock_operation::*;

            #[async_std::test]
            async fn buffer_access_awaits_completion() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (_, mock) = mock_operation(buffer.as_mut()).await;
                let a_ref = buffer.buffer_ref().await;
                assert_eq!(a_ref, &[0u8; 32] as &[u8]);
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

        mod notify_cancellation_intend {
            use super::super::super::*;
            use super::super::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_cancel() {
                ra_buffer_anchor!(buffer = [0u8; 32] of OpIntMock);
                let (mut op, mock) = mock_operation(buffer.as_mut()).await;
                mock.assert_not_run();
                op.notify_cancellation_intend().await;
                mock.assert_notify_cancel_run();
            }
        }

        mod cancellation {
            use super::super::super::*;
            use super::super::mock_operation::*;

            // we know this forwards so we only test if it forward to the right place
            #[async_std::test]
            async fn polls_op_int_poll_cancel() {
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