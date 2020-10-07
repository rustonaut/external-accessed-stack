use std::{mem, process::abort};

// Abort on panic in closure, useful for cases where panic recovery is not possible.
pub fn abort_on_panic<T>(func: impl FnOnce() -> T) -> T {
    let abort_on_drop = AbortOnDrop;
    let result = func();
    mem::forget(abort_on_drop);
    result
}


struct AbortOnDrop;

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        // with no_std we might need to implement abort by panic in panic.
        abort();
    }
}




