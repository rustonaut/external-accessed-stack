use core::task::{RawWaker, RawWakerVTable, Waker};

/// Returns a waker which does nothing on wake.
pub fn no_op_waker() -> Waker {
    let raw = raw_no_op_waker();
    //SAFE: The raw waker returned by `raw_no_op_waker` is always valid
    unsafe { Waker::from_raw(raw) }
}

/// Returns a valid raw waker which does nothing on wake.
///
/// # Safety
///
/// This must return a waker which always can be used with `Waker::from_raw`
/// in a safe way without any additional constraints.
pub fn raw_no_op_waker() -> RawWaker {
    RawWaker::new(0 as *const (), &NO_OP_WAKER_VTABLE)
}

/// VTable for a waker which does nothing on waker.
static NO_OP_WAKER_VTABLE: RawWakerVTable = {
    fn clone(_: *const ()) -> RawWaker { raw_no_op_waker() }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop_waker(_: *const ()) {}
    RawWakerVTable::new(clone, wake, wake_by_ref, drop_waker)
};
