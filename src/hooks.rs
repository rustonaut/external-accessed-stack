use core::{sync::atomic::spin_loop_hint, task::{Context, Poll}};

use crate::{define_atomic_hooks, op_int_utils::no_op_waker};


define_atomic_hooks! {
    /// The sync awaiting hook has unsafe constraints and is as such
    /// only exposed through a unsafe setter function and a safe
    /// getter function.
    ///
    static SYNC_AWAITING_HOOK: AtomicHook<DefaultSyncAwaitingHook: fn((OpaqueData, SyncAwaitingHookPollFn)) -> ()> = default_sync_awaiting_hook;
}

/// The type/function signature of a syn awaiting hook.
///
/// It is a function with two parameter:
///
/// 1. The opaque passed in data.
/// 2. A callback to call for polling
///
/// Due to limitations of the type system they are wrapped into a single tuple
/// parameter i.e. fn((data, pollfn)).
///
/// See [`replace_sync_awaiting_hook()`] for guarantees a hook implementation must give
/// on a rust-safety level.
///
pub type SynAwaitingHook = fn((OpaqueData, SyncAwaitingHookPollFn));

/// The type erased data passed through the hook to the poll
pub type OpaqueData = *mut ();

/// Function passed to the syn awaiting hook to poll for completion.
///
/// See [`replace_sync_awaiting_hook()`] for guarantees a hook implementation must give
/// on a rust-safety level.
pub type SyncAwaitingHookPollFn = fn(OpaqueData, &mut Context) -> Poll<()>;


/// Sets a new syn awaiting hook.
///
/// This can be used by operation interaction instances in the `make_sure_operation_ended()`
/// implementation to poll more efficient for completion.
///
/// Platforms can override this hook to inject thinks like yielding in the bussy loop or to
/// use a thread parker.
///
/// # Unsafe-Contract
///
/// The caller of this function must make sure the hook is correctly implemented. This entails that:
///
/// - The hook function must only return (in any way) from a call to it once the passed in poll fn did
///   return `Poll::Ready(())`. This is roughly the same guarantees implementors of `OperationInteraction.make_sure_operation_ended()`
///   need to uphold.
///
/// - The hook function must directly forward the passed in [`OpaqueData`] to the calls to the
///   given poll fn, it must not in any way touch or access the [`OpaqueData`] nor can it assume
///   anything about the validity of the pointer of the [`OpaqueData`].
///
/// - Both the passed in poll fn and opaque data pointer must only be used before this function
///   returns leaking them (e.g. using a `static` slot and using them afterwards is undefined
///   behavior. The opaque data pointer should be treated as if it's a reference of some kind
///   with a lifetime only valid for that call.
///
/// The most simple implementation is that of the `default_sync_await_hook` which does create
/// a `no_op_waker()` and then just busy polls the poll function. Any custom hook should do
/// something more advanced like using a parker or at least yielding the thread. But this
/// operations are platform dependent on `no_std`.
///
pub unsafe fn replace_sync_awaiting_hook(new_hook: SynAwaitingHook) -> SynAwaitingHook {
    SYNC_AWAITING_HOOK.replace(new_hook)
}

/// Returns the current syn awaiting hook.
///
/// # Default
///
/// ```
/// # use core::{sync::atomic::spin_loop_hint, task::{Context, Poll}};
/// # use remote_accessed_buffer::{op_int_utils::no_op_waker, hooks::*};
/// fn default_sync_await_hook((data, poll_fn): (OpaqueData, SyncAwaitingHookPollFn)) {
///     let waker = no_op_waker();
///     let mut ctx = Context::from_waker(&waker);
///     while let Poll::Pending = poll_fn(data, &mut ctx) {
///         spin_loop_hint();
///     }
/// }
/// ```
/// Which is the most basic viable implementation you can have, it's recommended to
/// override it with a more efficient platform/target dependent implementation.
///
/// # Safety
///
/// If implementors of `OperationInteraction` do want to use this to
/// implement `make_sure_operation_ended()`. They must make sure that
/// the passed in poll function only returns `Poll::Ready(())` once
/// the operation ended. If they do not do so then even after the
/// hook returns the operation might not have ended and further test
/// for completion will need to be added in the implementation of
/// `make_sure_operation_ended()`.
///
/// Furthermore implementations of `OperationInteraction` which use
/// the same poll function internally for the implementation of
///  `poll_completion` and the poll fn passed to this hook must make
/// sure the poll function is fused as it might already have returned
/// `Ready` in a awaiting of completion. But `make_sure_operation_ended()`
/// will always be called even if the polling for completion already
/// returned `Ready`!
pub fn get_sync_awaiting_hook() -> SynAwaitingHook {
    SYNC_AWAITING_HOOK.get()
}


/// Default implementation which just busy polls for completion.
//TODO: cfg if feature="std" use thread parking
fn default_sync_awaiting_hook((data, poll_fn): (OpaqueData, SyncAwaitingHookPollFn)) {
    let waker = no_op_waker();
    let mut ctx = Context::from_waker(&waker);
    while let Poll::Pending = poll_fn(data, &mut ctx) {
        spin_loop_hint();
    }
}

#[cfg(test)]
mod tests {

    mod default_sync_awaiting_hook {
        use super::super::*;

        #[test]
        fn polls_until_completion() {
            let mut counter = 10u8;

            default_sync_awaiting_hook((&mut counter as *mut _ as *mut (), poll_fn));

            assert_eq!(counter, 5);

            fn poll_fn(data: *mut (), _ctx: &mut Context) -> Poll<()> {
                //SAFE:
                //  - data is guaranteed to be passed through
                //  - poll_fn is guaranteed to only be called during the callback it was passed to
                let data: &mut u8 = unsafe { &mut *(data as *mut u8) };
                if *data > 5 {
                    *data -= 1;
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }
        }

    }
}