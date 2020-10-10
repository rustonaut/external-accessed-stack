use core::{marker::PhantomData, mem, pin::Pin, task::Poll, task::Waker};
use spin::Mutex;

#[derive(Debug)]
pub struct AtomicOperationCompleterAnchor<Result> {
    state: Inner<Result>,
    marker: PhantomData<Result>
}

impl<Result> AtomicOperationCompleterAnchor<Result> {

    /// Create a new anchor.
    ///
    /// The anchor will be pre-set in a `Waiting` state with no waker registered.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Waiting(None)),
            marker: PhantomData
        }
    }

    ///
    /// # Unsafe-Contract
    ///
    /// - Must only be called once.
    /// - The returned `AtomicOperationCompleter` must be used correctly.
    ///   Mainly it has a internal reference into the pinned self, but no lifetime.
    ///   Through the only way to invalidate the pointer target is by calling
    ///   [`AtomicOperationCompleter.complete_operation()`] which consumes it. Still
    ///   it must be made sure that the completer does not out-life the operation it
    ///   is made for.
    ///
    pub unsafe fn create_completer(self: Pin<&Self>) -> AtomicOperationCompleter<Result> {
        let inner = &self.as_ref().state as *const _;
        AtomicOperationCompleter { inner, marker: PhantomData }
    }

    /// Wait blocking for completion.
    ///
    /// If possible (i.e. you are in async code)
    /// prefer async calling poll instead.
    ///
    pub fn wait_for_completion(self: Pin<&Self>) {
        //TODO add a way to inject a thread parking mechanism,
        //     but as we don't know if there are threads we
        //     can't implement it instead we can add a `AtomicUsize/Ptr`
        //     based function hook.
        let state = &self.as_ref().state;
        loop {
            //This loop is a situation where spin::Mutex **might** not be the best choice
            //but something which always gives priority to the completion locking the call.
            let state = state.lock();
            match &*state {
                State::Completed(_) =>  { break; }
                _ => {}
            }
            //TODO some kind of sleep/yield function
            core::sync::atomic::spin_loop_hint();
        }
    }

    /// Polls for completion.
    ///
    /// This might update the current waker.
    ///
    /// Due to aliasing rules this must be based on `Pin<&Self>` as
    /// we can go from a `&Self` to a `&mut` using a `UnsafeCell` but
    /// the other way around is currently not possible in rust.
    ///
    /// # Panic
    ///
    /// Panics if polled after it returned `Ready` (i.e. polled
    /// after we already did communicate completion).
    pub fn poll(self: Pin<&Self>, waker: &Waker) -> Poll<Result> {
        use self::State::*;

        let state = &self.as_ref().state;
        let mut state = state.lock();

        let mut replacement_state;

        match &*state {
            Waiting(Some(current_waker)) => {
                if waker.will_wake(current_waker) {
                    return Poll::Pending;
                }
                replacement_state = Waiting(Some(waker.clone()));
            },
            Waiting(None) => {
                replacement_state = Waiting(Some(waker.clone()));
            },
            Completed(Some(_)) => {
                replacement_state = Completed(None);
            },
            Completed(None) => {
                // make sure to only panic after releasing the lock
                replacement_state = Completed(None);
            }
        }

        mem::swap(&mut *state, &mut replacement_state);
        drop(state);

        // match **previous state**
        match replacement_state {
            Completed(Some(value)) => {
                return Poll::Ready(value);
            },
            Completed(None) => {
                panic!("polled after returning Poll::Ready")
            },
            _ => {
                // Completed(None) => unreachable
                // Waiting(Some(_)) => pending - no need to wake the old one
                //                               engines which need that should
                //                               implement wake on drop anyway
                // Waiting(None) => pending
                return Poll::Pending;
            }
        }
    }
}

//TODO: Consider adding a "abort on hang" option which aborts on
//      .make_sure_operation_completed() calls if the Completer was
//      dropped without calling .complete_operation().
//
//      We could have a `AtomicUsize`/`AtomicPtr` containing a function
//      which is called on "async_hang_detected()" and called on
//      "sync_hang_detected()".
#[derive(Debug)]
pub struct AtomicOperationCompleter<Result> {
    inner: *const Inner<Result>,
    marker: PhantomData<Result>
}

unsafe impl<Result: Send> Send for AtomicOperationCompleter<Result> {}
unsafe impl<Result: Sync> Sync for AtomicOperationCompleter<Result> {}

impl<Result> AtomicOperationCompleter<Result> {

    ///
    /// # Unsafe-Contract
    ///
    /// Caller must make sure that before this call:
    ///
    /// - The operation this is used for completed.
    /// - There is no longer any form of reference to the buffer. A pointer
    ///   if not used is ok. But no `&`, `&mut` or similar must exists anymore nor must
    ///   a pointer be dereferenced anymore. Be aware that literally a existing `&`,`&mut`
    ///   is already a violation of this contract even if it is not used!
    pub unsafe fn complete_operation(self, result: Result) {
        use self::State::*;

        //Safe: We know the pointer target is still valid.
        let mut state = (&*self.inner).lock();
        let prev_state = mem::replace(&mut *state, State::Completed(Some(result)));
        drop(state);
        mem::forget(self);

        if let Waiting(Some(waker)) = prev_state {
            waker.wake()
        }
    }
}


type Inner<Result> = Mutex<State<Result>>;
#[derive(Debug)]
enum State<Result> {
    Waiting(Option<Waker>),
    Completed(Option<Result>)
}



#[cfg(test)]
mod tests {
    use super::*;
    use core::{sync::atomic::{AtomicUsize, Ordering}, time::Duration};

    use std::{sync::Arc, thread};
    use pin_utils::pin_mut;
    use waker_fn::waker_fn;

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct OpaqueResult(u16);

    #[test]
    fn completion_sets_result() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };

        let waker = waker_fn(|| {});
        let res = anchor.as_ref().poll(&waker);
        assert_eq!(res, Poll::Pending);

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()) };

        let res = anchor.as_ref().poll(&waker);
        assert_eq!(res, Poll::Ready(result));
    }

    #[test]
    fn waker_are_registered_and_called() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };

        let cn = Arc::new(AtomicUsize::new(0));
        let waker = waker_fn({let cn = cn.clone(); move || { cn.fetch_add(1, Ordering::SeqCst); } });

        let res = anchor.as_ref().poll(&waker);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn.load(Ordering::SeqCst), 0);

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()); }
        assert_eq!(cn.load(Ordering::SeqCst), 1);

        let res = anchor.as_ref().poll(&waker);
        assert_eq!(res, Poll::Ready(result));
        assert_eq!(cn.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn wakers_can_be_replaced() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };

        let cn1 = Arc::new(AtomicUsize::new(0));
        let waker1 = waker_fn({let cn1 = cn1.clone(); move || { cn1.fetch_add(1, Ordering::SeqCst); } });
        let cn2 = Arc::new(AtomicUsize::new(0));
        let waker2 = waker_fn({let cn2 = cn2.clone(); move || { cn2.fetch_add(1, Ordering::SeqCst); } });

        let res = anchor.as_ref().poll(&waker1);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 0);

        let res = anchor.as_ref().poll(&waker2);
        assert_eq!(res, Poll::Pending);
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 0);

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()); }
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 1);


        let waker3 = waker_fn(|| panic!("Poll leading to Ready should not wake waker."));
        let res = anchor.as_ref().poll(&waker3);
        assert_eq!(res, Poll::Ready(result));
        assert_eq!(cn1.load(Ordering::SeqCst), 0);
        assert_eq!(cn2.load(Ordering::SeqCst), 1);
    }

    #[should_panic(expected = "polled after returning Poll::Ready")]
    #[test]
    fn calling_poll_after_ready_panics() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };
        let waker = waker_fn(|| {});

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()); }

        let _ = anchor.as_ref().poll(&waker);
        let _ = anchor.as_ref().poll(&waker);
    }

    #[test]
    fn calling_completion_before_first_poll_works() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };
        let waker = waker_fn(|| {});

        let result = OpaqueResult::default();
        unsafe { completer.complete_operation(result.clone()); }

        let res = anchor.as_ref().poll(&waker);
        assert_eq!(res, Poll::Ready(result));
    }

    #[test]
    fn works_across_thread_boundaries() {
        let anchor = AtomicOperationCompleterAnchor::<OpaqueResult>::new();
        pin_mut!(anchor);
        let completer = unsafe { anchor.as_ref().create_completer() };
        let waker = waker_fn(|| {});

        let result = OpaqueResult::default();
        let join = thread::spawn({
            let result = result.clone();
            move || {
                thread::sleep(Duration::from_millis(10));
                unsafe { completer.complete_operation(result); }
            }
        });

        loop {
            if let Poll::Ready(recv_result) = anchor.as_ref().poll(&waker) {
                assert_eq!(result, recv_result);
                break;
            }
        };


        let _ = join.join();
    }
}