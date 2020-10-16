use core::mem;


// Abort on panic in closure, useful for cases where panic recovery is not possible.
pub fn abort_on_panic<T>(func: impl FnOnce() -> T) -> T {
    let abort_on_drop = AbortOnDrop;
    let result = func();
    mem::forget(abort_on_drop);
    result
}

/// Implementation of abort by using panic during panic.
///
/// For `no_std` we can't use `std::process::abort`.
fn abort() -> ! {
    let _cause_abort = PanicOnDrop;
    panic!("First panic to cause abort");
}

struct PanicOnDrop;

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!("panic on drop (likely abort through double panic)")
    }
}


struct AbortOnDrop;

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        // with no_std we might need to implement abort by panic in panic.
        abort();
    }
}




/// A more readable `!` operator
#[cfg(any(test, feature="op-int-utils"))]
#[inline(always)]
pub fn not(val: bool) -> bool {
    !val
}