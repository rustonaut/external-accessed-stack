use std::{sync::atomic::AtomicUsize, any::TypeId, cell::UnsafeCell, pin::Pin};

use futures_lite::future::block_on;
use external_accessed_stack::{*, op_int_utils::atomic_state};
use fake_dma::{Singletons, create_singletons, Stream, Periphery, Transfer, S1, FooBarPeriphery};

mod fake_dma;


#[test]
fn use_case_emulation() {
    block_on(async {
        unsafe { fake_dma::S1_INTERRUPT = interrupt_s1::completion_interrupt };
        let Singletons { s1, foo_bar_periphery:mut peri } = create_singletons();

        ea_stack_value!(buffer: StackValueHandle<[u8], DMAAnchor<S1, FooBarPeriphery>> =  [0u8; 32]);

        peri.replace_data(vec![23; 30]);

        let handle = start_copy_data_to_buffer(buffer.reborrow(), s1, peri).await;
        // to avoid the closure buffer.access_value_after_completion and buffer.try_get_buffer_mut()
        // and similar can be used, see below.
        let (s1, mut peri) = handle.access_value_after_completion(|buffer_mut, res| {
            for byte in &buffer_mut[..30] {
                assert_eq!(*byte, 23);
            }
            for byte in &buffer_mut[30..] {
                assert_eq!(*byte, 0);
            }
            res
        }).await;

        peri.replace_data(vec![42; 32]);
        start_copy_data_to_buffer(buffer.reborrow(), s1, peri).await;

        let (buffer_mut, _opt_res) = buffer.access_value_after_completion().await;
        for byte in &buffer_mut[..] {
            assert_eq!(*byte, 42);
        }
    });
}

pub async fn start_copy_data_to_buffer<'a, S,P>(
    mut buffer: StackValueHandle<'a, [u8], DMAAnchor<S, P>>,
    stream: S,
    peri: P
) -> OperationHandle<'a, [u8], DMAAnchor<S, P>>
where
    S: Stream + Send,
    P: Periphery + Send
{
    buffer.cancellation().await;

    //FIXME: probably some more considerations about panics
    let dma_buffer = unsafe { buffer.try_get_unsafe_embedded_dma_buffer().unwrap() };
    let transfer = Transfer::init(stream, peri, dma_buffer);
    let anchor = DMAAnchor::new(transfer);
    //SAFE[UNWRAP]: We awaited cancellation
    let op_hdl = unsafe { buffer.try_register_new_operation(anchor).unwrap() };
    //SAFE: Guaranteed to be the first time we called this, we didn't start the transfer type
    unsafe { op_hdl.operation_interaction().unwrap().start() };
    op_hdl
}


pub struct DMAAnchor<S, P>
where
    S: Stream + Send + 'static,
    P: Periphery + Send
{
    inner: atomic_state::Anchor<()>,
    singletons: UnsafeCell<Option<Transfer<S, P, UnsafeEmbeddedDmaBuffer<u8>>>>
}

impl<S, P> DMAAnchor<S,P>
where
    S: Stream + Send + 'static,
    P: Periphery + Send
{
    fn new(transfer: Transfer<S, P, UnsafeEmbeddedDmaBuffer<u8>>) -> Self {
        Self {
            inner: atomic_state::Anchor::new(),
            singletons: UnsafeCell::new(Some(transfer))
        }
    }

    fn inner(self: Pin<&Self>) -> Pin<&atomic_state::Anchor<()>> {
        //SAFE: E.g. see project-pin
        return unsafe { self.map_unchecked(inner) };
        //for some reason inference dies on map_unchecked
        fn inner<S2,P2>(s: &DMAAnchor<S2,P2>) -> &atomic_state::Anchor<()> where S2: Stream + Send + 'static, P2: Periphery + Send { &s.inner}
    }

    /// # Safety
    ///
    /// Call only once and `Transfer` must not yet be started!
    unsafe fn start(self: Pin<&Self>) {
        let completer = self.inner().create_completer();
        set_completer::<S>(completer);
        (&mut *self.singletons.get())
            .as_mut()
            .unwrap() //SAFE[Unwrap]: Only becomes None on completion
            .start()
    }
}

unsafe impl<S,P> OperationInteraction for DMAAnchor<S, P>
where
    S: Stream + Send + 'static,
    P: Periphery + Send
{
    type Result = (S, P);

    fn make_sure_operation_ended(self: std::pin::Pin<&Self>) {
        self.inner().make_sure_operation_ended()
    }

    fn poll_completion(self: std::pin::Pin<&Self>, cx: &mut std::task::Context) -> std::task::Poll<Self::Result> {
        self.inner().poll_completion(cx).map(|()| {
            //SAFE: This branch will only rune once, as `inner.poll_completion` panics if polled after returning Ready
            //SAFE[UNWRAP]: For same reason as SAFE
            let transfer = unsafe { (&mut * self.singletons.get()).take().unwrap() };
            let (stream, peri, _buf) = transfer.free();
            (stream, peri)
        })
    }

    fn poll_request_cancellation(self: std::pin::Pin<&Self>, _cx: &mut std::task::Context) -> std::task::Poll<()> {
        //Omitted: stream.disable()
        std::task::Poll::Ready(())
    }
}


fn set_completer<S>(completer: atomic_state::Completer<()>)
where
    S: Stream + 'static,
{
    if TypeId::of::<S>() == TypeId::of::<fake_dma::S1>() {
        interrupt_s1::set_completer(completer)
            .unwrap_or_else(|_| panic!("completer setting and operation tracking got out of sync"));
    } else {
        unreachable!("We only have S1")
    }
}


mod interrupt_s1 {
    use super::*;
    use atomic_state::Completer;
    use std::sync::atomic::Ordering;

    static COMPLETER_SLOT: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn set_completer(completer: Completer<()>) -> Result<(), Completer<()>> {
        let completer = completer.into_usize();
        if let Ok(_) = COMPLETER_SLOT.compare_exchange(0, completer, Ordering::Release, Ordering::Relaxed) {
            Ok(())
        } else {
            Err(unsafe { Completer::<()>::from_usize(completer)})
        }
    }

    pub(super) fn completion_interrupt() {
        //Omitted: if stream.get_completion_flag() != true { return; }
        let completer = COMPLETER_SLOT.swap(0, Ordering::AcqRel);
        if completer != 0 {
            //SAFE: We only put valid completers and 0 in there and never do a load
            //      which could have duplicated the completer.
            let completer = unsafe { Completer::<()>::from_usize(completer) };
            //SAFE: Interrupt is only called after completion.
            //      WARNING: This is true for this example in the real code a check for
            //               completion needs to be done.
            unsafe { completer.complete_operation(()) };
        }
    }
}



