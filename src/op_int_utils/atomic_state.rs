//! This module containing a atomic state which can be used by a `OperationInteraction` implementation.
//!
//! It mainly contains two types:
//!
//! 1. [`Anchor`] which should be wrapped by an [`OperationInteraction`] implementation and which
//!    does itself implement [`OperationInteraction`]. This type is moved into the `EABufferAnchor`
//!    when registering a new operation. It's the pinned anchor we can point to.
//!
//! 2. [`Completer`] which is a handle which can be used to mark a operation as completed and send
//!    back the result. This one can be created from the `Anchor` after it's pinned inside the
//!    `EABufferAnchor`. In this implementation `Completer` is not `Sync` (but `Send`) so only
//!    one thread (or similar) can complete the operation.
//!
use crate::{hooks::get_sync_awaiting_hook, utils::not, OperationInteraction};
use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    pin::Pin,
    ptr,
    sync::atomic::AtomicU8,
    sync::atomic::Ordering,
    task::Context,
    task::Poll,
    task::Waker,
};

/// Flags used for the atomic state machine.
mod flag {
    /// If the the operation completed
    pub(super) const COMPLETED: u8 = 1 << 0;

    /// If set the anchor is currently updating the waker
    pub(super) const UPDATE_WAKER: u8 = 1 << 1;

    /// To keep track of weather or not we did hand out a `Completer`.
    pub(super) const INITIALIZED: u8 = 1 << 2;

    #[inline(always)]
    pub(super) fn is_set(state: u8, flag: u8) -> bool {
        (state & flag) != 0
    }
}

/// A utility which can be used to implement a [`OperationInteraction`].
///
/// While this does implement [`OperationInteraction`] it is recommended to
/// wrap it into a platform specific interaction to prevent any accidental
/// ossification.
///
/// While this type on itself is compatible with interrupt (by being lock-free)
/// you can only use it to complete operations from interrupts *if calling
/// [`Waker.wake()`] from an interrupt is safe*. (If not you could still have something
/// like a lock-free queue to which you push the waker on an interrupt and then the
/// queue consuming thread/task calls [`Waker.wake()`]. But that has probably some
/// noticeable drawbacks.
///
/// When ignoring "spurious/external induced" compare exchange failure on platforms
/// like ARM this algorithm is wait-free. When not ignoring it it can not be wait free
/// as as (strong) `compare_exchange` is not wait free on ARM, (e.g. due to some interrupts)
/// the compare_exchange operation might (theoretically) never succeed.
/// But this is only theoretically relevant I think.
///
/// # Drop
///
/// Dropping it will poll for completion if a completer had been handed out.
#[derive(Debug)]
pub struct Anchor<Result> {
    inner: InnerAnchor<Result>,
    /// Marker to prevent `Sync`/`Send`
    not_sync_send: PhantomData<*const ()>,
}

/// The `AtomicStateAnchor` impl.
///
/// This is neccessary to make sure the
/// `AtomicStateAnchor` is `!Sync` and `!Send` while having a inner state
/// which can be `Synced` to exactly one single place (the single completer
/// which is handed out).
#[derive(Debug)]
struct InnerAnchor<Result> {
    /// The state used to synchronize access to the waker_slot and result_slot.
    state_sync: AtomicU8,
    /// Until the [`flag::COMPLETED`] flag is set this is "owned" by the anchor.
    ///
    /// While it's owned by the anchor the anchor can write new wakers to it
    /// mut must "guard" the writes by setting the [`flag::UPDATE_WAKER`] flag.
    ///
    /// Once the [`flag::COMPLETED`] flag is set [`flag::UPDATE_WAKER`] must no longer be set.
    ///
    /// Once the [`flag::COMPLETED`] flag is set it's "owned" by completer *if and
    /// only if* the [`flag::UPDATE_WAKER`] flag is **not** set!
    ///
    /// If the [`flag::COMPLETED`] flag is set while un-setting the [`flag::UPDATE_WAKER`] flag
    /// the anchor must first take out any waker before then un-setting the
    /// [`flag::UPDATE_WAKER`] flag.
    waker_slot: UnsafeCell<Option<Waker>>,
    /// The slot for passing back the result.
    ///
    /// Until the [`flag::COMPLETED`] flag is set this is "owned" by the completer,
    /// afterwards it's owned by the anchor.
    ///
    /// The completer will write the result and then set the flag, after
    /// this the anchor can read the result.
    ///
    result_slot: UnsafeCell<Option<Result>>,
}

/// SAFE: We do guard access with the sate_sync atomic, waker is Sync anyway.
///       The way we use the cells doesn't allow multiple `&` borrows from different
///       threads at the same time so `Result` only needs to be send.
unsafe impl<Result> Sync for InnerAnchor<Result> where Result: Send {}

impl<Result> Anchor<Result> {
    /// Create a new anchor.
    ///
    /// The anchor will be pre-set in a `Waiting` state with no waker registered.
    pub fn new() -> Self {
        Self {
            inner: InnerAnchor {
                state_sync: AtomicU8::new(0),
                waker_slot: UnsafeCell::new(None),
                result_slot: UnsafeCell::new(None),
            },
            not_sync_send: PhantomData,
        }
    }

    /// Creates a completer, this must only be called once.
    ///
    /// `Pin` guarantees enforce that once the anchor is pinned
    /// it must no longer be moved until dropped. Which means you
    /// MUST create the completer *after* moving the anchor into
    /// the underlying `EABufferAnchor`. Everything else would require
    /// a unsafe-contract violation (of e.g. [`Pin`] or [`Completer.complete_operation`]).
    ///
    /// # Panic
    ///
    /// If this is called twice this will panic. Depending
    /// on where this is panics this might have a high chance
    /// to trigger an abort.
    ///
    pub fn create_completer(self: Pin<&Self>) -> Completer<Result> {
        if let Err(state) = self.inner.state_sync.compare_exchange(
            0,
            flag::INITIALIZED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            panic!("Unexpected state when trying to create completer: {:b}\nCompleter must only be created once.", state);
        }

        let ptr = &self.inner as *const _;
        Completer { ptr }
    }

    /// Polls to check if the operation completed.
    ///
    /// Polling to check if the operation completed includes updating
    /// the `Waker`.
    pub fn poll(&self, waker: &Waker) -> Poll<Option<Result>> {
        // Self is !Sync so this can not be called concurrently.
        //
        // We will:
        // 1. set the UPDATE_WAKER flag
        // 2. this can only fail if the `COMPLETED` flag was set, in which case
        //    we now own the `result_slot` and can return a result.
        // 3. if this doesn't fail we own the `waker_slot` until we un-set the
        //    `UPDATE_WAKER` flag, so we can write the waker.
        // 4. the we un-set the `UPDATE_WAKER` flag
        // 5. If the operation completed since then this fails and we still own the
        //    waker, so now we need to take out the waker and the result

        let result = self.inner.state_sync.compare_exchange(
            flag::INITIALIZED,
            flag::INITIALIZED | flag::UPDATE_WAKER,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        if let Err(state) = result {
            if flag::is_set(state, flag::COMPLETED) {
                //SAFE: The `COMPLETED` flag is set
                return Poll::Ready(unsafe { self.extract_result() });
            }
            panic!("Unexpected state when setting UPDATE_WAKER: {:b}", state);
        }

        // now we can write to the `waker_slot`
        //SAFE: The `UPDATE_WAKER` flag is set.
        let old_waker = unsafe { self.update_waker(waker) };

        // unset the UPDATE_WAKER flag
        let result = self.inner.state_sync.compare_exchange(
            flag::INITIALIZED | flag::UPDATE_WAKER,
            flag::INITIALIZED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        // only drop old waker after un-setting the flag
        // to slightly decrease the risc of an unnecessary abort
        drop(old_waker);

        if let Err(state) = result {
            if flag::is_set(state, flag::COMPLETED) {
                // SAFE: The `COMPLETED` flag is set.
                return Poll::Ready(unsafe { self.extract_result() });
            }
            panic!("Unexpected state when un-setting UPDATE_WAKER: {:b}", state);
        }

        Poll::Pending
    }

    /// Updates the waker if necessary.
    ///
    /// Returns the old waker if there was any and it
    /// did update the waker.
    ///
    /// # Unsafe-Contract
    ///
    /// Must only be called if the `UPDATE_WAKER` flag is set.
    unsafe fn update_waker(&self, waker: &Waker) -> Option<Waker> {
        let slot = &mut *self.inner.waker_slot.get();
        let waker_needs_update = slot
            .as_ref()
            .map(|old_waker| !old_waker.will_wake(waker))
            .unwrap_or(true);

        if waker_needs_update {
            mem::replace(slot, Some(waker.clone()))
        } else {
            None
        }
    }

    /// Extracts the result.
    ///
    /// Returns `None` if the result was already extracted beforehand.
    ///
    /// (Or was not yet set, but in that case you broke the unsafe contract
    /// of this method by calling it before `COMPLETED` was set).
    ///
    /// # Unsafe-Contract
    ///
    /// Must only be called if the `COMPLETED` flag is set.
    unsafe fn extract_result(&self) -> Option<Result> {
        ptr::replace(self.inner.result_slot.get(), None)
    }
}

impl<Result> Default for Anchor<Result> {
    fn default() -> Self {
        Self::new()
    }
}

//SAFE: As long as the unsafe contract of the [`Completer.complete()`] method is uphold
//      this is a safe implementation.
unsafe impl<Result> OperationInteraction for Anchor<Result> {
    type Result = Result;

    fn make_sure_operation_ended(self: Pin<&Self>) {
        let self_as_opaque_data = self.get_ref() as *const Self as *mut ();
        let sync_awaiting = get_sync_awaiting_hook();

        sync_awaiting((self_as_opaque_data, callback::<Result>));

        fn callback<R>(data: *mut (), ctx: &mut Context) -> Poll<()> {
            //SAFE: We have the guarantees that data was not touched.
            let self_ = unsafe { &*(data as *const _ as *const Anchor<R>) };
            //SAFE: recreates Pin which we know exists
            let self_ = unsafe { Pin::new_unchecked(self_) };
            self_.poll(ctx.waker()).map(|_res| ())
        }
    }

    fn poll_completion(self: Pin<&Self>, cx: &mut core::task::Context) -> Poll<Self::Result> {
        self.poll(cx.waker()).map(|opt_result| {
            opt_result.expect("Polled `poll_completion` after it returned `Ready`")
        })
    }
}
/// A [`Completer`] for a given `op_int_utils::atomic_state::Anchor`.
///
/// For any [`Anchor`] there is always only at most one completer.
///
/// The [`Completer`] is `Send` but not `Sync` which means only
/// one thread (or similar) can complete it.
///
/// Only the completer can mark a operation as completed and send
/// back the result.
///
/// Completing an operation is unsafe as you must guarantee that
/// it's actually completed and no more access to the underlying
/// buffer is done.
///
#[derive(Debug)]
pub struct Completer<Result> {
    ptr: *const InnerAnchor<Result>,
}

/// SAFE: If the Result is `Send` InnerAnchor is `Sync` and we can `Send` the
///       handle to another thread to then send the result back.
unsafe impl<Result> Send for Completer<Result> where Result: Send {}

impl<Result> Completer<Result> {
    /// Marks the operation as completed and sends back the result.
    ///
    /// After marking the operation as completed this will call `Waker.wake()`
    /// if necessary. Use [`Completer.complete_operation_no_wake()`] to have
    /// a variant of this method which doesn't wake the waker.
    ///
    /// # Safety
    ///
    /// Caller must make sure that before this call:
    ///
    /// - The operation this is used for completed.
    /// - There is no longer any form of reference to the buffer. A pointer
    ///   if not used is ok. But no `&`, `&mut` or similar must exists anymore nor must
    ///   a pointer be dereferenced anymore. Be aware that literally a existing `&`,`&mut`
    ///   is already a violation of this contract even if it is not used!
    /// - The anchor has not been dropped. If the completer was created through a `Anchor`
    ///   pinned into a `EABufferAnchor` this is already enforced.
    /// - This is only called once (which is implicitly enforced by consuming self).
    ///
    pub unsafe fn complete_operation(self, result: Result) {
        if let Some(waker) = self.complete_operation_no_wake(result) {
            waker.wake()
        }
    }

    /// See [`Completer.complete_operation()`].
    ///
    /// # Safety
    ///
    /// The unsafe-contract is the exact same as [`Completer.complete_operation()`].
    ///
    pub unsafe fn complete_operation_no_wake(self, result: Result) -> Option<Waker> {
        // We do following:
        // 1. As we are the only ones which can set the COMPLETED flag we
        //    and this method can only be called once per anchor we know that
        //    we still own the `result_slot` and can as such write a result.
        // 2. Then we set the `COMPLETED` flag.
        // 3. If the `UPDATE_WAKER` flag was set when setting the `COMPLETED` flag we
        //    know that the polling task is awake and will notice the completion before
        //    going to sleep so we don't need to do anything.
        // 4. If the `UPDATE_WAKER` flag is not set we know we now own the `waker_slot` and
        //    as it won't accessed by the polling task anymore. So we take out the waker and
        //    return it.

        // Make sure not to drop self.
        let self_ = ManuallyDrop::new(self);
        //SAFE: By the outer unsafe contract this ptr must still be valid and properly aliased.
        let anchor = &*self_.ptr;
        //SAFE: By the state machine we know we must currently own the slot
        ptr::replace(anchor.result_slot.get(), Some(result));
        // We need AcqRel here as we release the COMPLETED flag and acquire if we do now own
        // the waker_slot.
        let state = anchor
            .state_sync
            .fetch_or(flag::COMPLETED, Ordering::AcqRel);
        if not(flag::is_set(state, flag::UPDATE_WAKER)) {
            // SAFE: As the polling task wasn't currently accessing the waker slot we now own it
            ptr::replace(anchor.waker_slot.get(), None)
        } else {
            None
        }
    }

    /// Turns the completer into a usize.
    ///
    /// Warning: This  must not be used to clone the completer, this is just a utility to
    ///          allow it to be used with `AtomicUsize`
    pub fn into_usize(self) -> usize {
        let Self { ptr } = self;
        ptr as usize
    }

    /// Creates a completer from a usize created by `Self.into_usize()`.
    ///
    /// # Safety
    ///
    /// 1. The usize must have been created with [`Completer.into_usize()`],
    ///    note that the **exact** same `Self` type must have been used. I.e.
    ///    having the same `Result` type.
    /// 2. This must not be used to create copies of the completer, the caller
    ///    must guarantee that for a given completer which was turned into a usize
    ///    only at most one "re-created" completer exist at any point in time.
    ///
    pub unsafe fn from_usize(ptr: usize) -> Self {
        let ptr = mem::transmute(ptr);
        Completer { ptr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        sync::atomic::{AtomicUsize, Ordering},
        task::Context,
        time::Duration,
    };

    use pin_utils::pin_mut;
    use std::{sync::Arc, thread};
    use waker_fn::waker_fn;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct OpaqueResult(u16);

    #[test]
    fn completion_sets_result() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();
        let waker = waker_fn(|| {});
        let mut ctx = Context::from_waker(&waker);

        let res = anchor.as_ref().poll_completion(&mut ctx);
        assert_eq!(res, Poll::Pending);

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()) };

        let res = anchor.as_ref().poll_completion(&mut ctx);
        assert_eq!(res, Poll::Ready(result));
    }

    #[test]
    fn waker_are_registered_and_called() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();

        let cn = Arc::new(AtomicUsize::new(0));
        let waker = waker_fn({
            let cn = cn.clone();
            move || {
                cn.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut ctx = Context::from_waker(&waker);

        let res = anchor.as_ref().poll_completion(&mut ctx);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn.load(Ordering::SeqCst), 0);

        let result = OpaqueResult::default();
        unsafe {
            completer.complete_operation(result.clone());
        }
        assert_eq!(cn.load(Ordering::SeqCst), 1);

        let res = anchor.as_ref().poll_completion(&mut ctx);
        assert_eq!(res, Poll::Ready(result));
        assert_eq!(cn.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn wakers_can_be_replaced() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();

        let cn1 = Arc::new(AtomicUsize::new(0));
        let waker1 = waker_fn({
            let cn1 = cn1.clone();
            move || {
                cn1.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut ctx1 = Context::from_waker(&waker1);
        let cn2 = Arc::new(AtomicUsize::new(0));
        let waker2 = waker_fn({
            let cn2 = cn2.clone();
            move || {
                cn2.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut ctx2 = Context::from_waker(&waker2);

        let res = anchor.as_ref().poll_completion(&mut ctx1);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 0);

        let res = anchor.as_ref().poll_completion(&mut ctx2);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 0);

        let result = OpaqueResult::default();
        unsafe {
            completer.complete_operation(result.clone());
        }
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 1);

        let waker3 = waker_fn(|| panic!("Poll leading to Ready should not wake waker."));
        let mut ctx3 = Context::from_waker(&waker3);
        let res = anchor.as_ref().poll_completion(&mut ctx3);
        assert_eq!(res, Poll::Ready(result));
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 1);
    }

    #[should_panic(expected = "Polled `poll_completion` after it returned `Ready`")]
    #[test]
    fn calling_poll_after_ready_panics() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();
        let waker = waker_fn(|| {});
        let mut ctx = Context::from_waker(&waker);

        let result = OpaqueResult::default();
        unsafe {
            completer.complete_operation(result.clone());
        }

        let _ = anchor.as_ref().poll_completion(&mut ctx);
        let _ = anchor.as_ref().poll_completion(&mut ctx);
    }

    #[test]
    fn calling_completion_before_first_poll_works() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();
        let waker = waker_fn(|| {});
        let mut ctx = Context::from_waker(&waker);

        let result = OpaqueResult::default();
        unsafe {
            completer.complete_operation(result.clone());
        }

        let res = anchor.as_ref().poll_completion(&mut ctx);
        assert_eq!(res, Poll::Ready(result));
    }

    #[test]
    fn works_across_thread_boundaries() {
        let anchor = Anchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = anchor.as_ref().create_completer();
        let waker = waker_fn(|| {});
        let mut ctx = Context::from_waker(&waker);

        let result = OpaqueResult::default();
        let join = thread::spawn({
            let result = result.clone();
            move || {
                thread::sleep(Duration::from_millis(10));
                unsafe {
                    completer.complete_operation(result);
                }
            }
        });

        loop {
            if let Poll::Ready(recv_result) = anchor.as_ref().poll_completion(&mut ctx) {
                assert_eq!(result, recv_result);
                break;
            }
        }

        let _ = join.join();
    }
}
