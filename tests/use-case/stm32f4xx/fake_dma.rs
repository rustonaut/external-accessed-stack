//! Roughly emulates the DMA interface provided by the stm32f4xx-hal crate.
//!
use embedded_dma::{WriteBuffer, ReadBuffer};
use std::{ptr, sync::atomic::{AtomicUsize, AtomicBool, Ordering}, cmp::min, mem, thread::{self, JoinHandle}};

/// Stream Singleton
pub trait Stream {
    fn clear_interrupt_flags(&self);
    fn __completion_callback(&self) -> fn();
}

pub static S1_FLAGS: AtomicUsize = AtomicUsize::new(0);
pub static mut S1_INTERRUPT: fn() = || {};

pub struct S1 { _priv: () }
impl Stream for S1 {
    fn clear_interrupt_flags(&self) {
        S1_FLAGS.store(0, Ordering::Release);
    }

    fn __completion_callback(&self) -> fn() {
       || {
           S1_FLAGS.fetch_or(0b1, Ordering::Release);
           unsafe { S1_INTERRUPT() }
       }
    }
}


pub trait Periphery: ReadBuffer<Word=u8> {}

pub struct FooBarPeriphery { data: Vec<u8> }

impl FooBarPeriphery {

    fn new() -> Self {
        FooBarPeriphery { data: Vec::new() }
    }

    pub fn replace_data(&mut self, data: Vec<u8>) -> Vec<u8> {
        mem::replace(&mut self.data, data)
    }
}

impl Periphery for FooBarPeriphery {}
unsafe impl ReadBuffer for FooBarPeriphery {
    type Word = u8;

    unsafe fn read_buffer(&self) -> (*const Self::Word, usize) {
        let ptr = self.data.as_ptr();
        let len = self.data.len();
        (ptr, len)
    }
}


pub struct Singletons {
    pub s1: S1,
    pub foo_bar_periphery: FooBarPeriphery
}


pub fn create_singletons() -> Singletons {
    static SINGLETONS_CREATED: AtomicBool = AtomicBool::new(false);
    let was_initialized = SINGLETONS_CREATED.swap(true, Ordering::AcqRel);

    if was_initialized {
        panic!("create_singletons() called twice");
    }

    Singletons {
        s1: S1 { _priv: () },
        foo_bar_periphery: FooBarPeriphery::new()
    }
}


/// Emulates a simplified version of DMA interface exposed by the stm32f4xx-hal crate.
///
/// This is simplified in following way:
///
/// - Not emulating any of the low level details, as it's unnecessary to show if this
///   buffer can be used with such an interface. This includes not having a config.
///
/// - Consumes less singleton resources, the original api has following singletons (by trait):
///   `Stream`, `Periphery`. As well a a `Channel` and a `DmaDirection` generic which is not a
///   singleton. But this emulated interface only uses `Stream` and a `Periphery` singleton as
///   that should be good enough to emulate the usability.
///
pub struct Transfer<STREAM, PERIPHERY, BUF>
where
    STREAM: Stream + Send,
    PERIPHERY: Periphery + Send,
    BUF: WriteBuffer<Word=u8> + Send + 'static
{
    stream: STREAM,
    periphery: PERIPHERY,
    buffer: BUF,
    thread: Option<JoinHandle<()>>
}

impl<STREAM, PERIPHERY, BUF> Transfer<STREAM, PERIPHERY, BUF>
where
    STREAM: Stream + Send,
    PERIPHERY: Periphery + Send, //embedded-dma should have this bounds too, I think.
    BUF: WriteBuffer<Word=u8> + Send + 'static
{
    //apply_config is not implemented as not necessary for this test


    pub fn init(
        stream: STREAM,
        periphery: PERIPHERY,
        buffer: BUF
    ) -> Self {
        Self {
            stream,
            periphery,
            buffer,
            thread: None
        }
    }

    pub fn start(&mut self) {
        self.__await_completion();
        self.stream.clear_interrupt_flags();
        let completion_callback = self.stream.__completion_callback();
        let (read_ptr, read_len) =  unsafe { self.periphery.read_buffer() };
        let (write_ptr, write_len) = unsafe { self.buffer.write_buffer() };
        let len = min(read_len, write_len);
        let data = SendPackage { read_ptr, write_ptr, len };

        self.thread = Some(thread::spawn(move || {
            unsafe { ptr::copy(data.read_ptr, data.write_ptr, data.len) };
            completion_callback()
        }));

        struct SendPackage {
            read_ptr: *const u8,
            write_ptr: *mut u8,
            len: usize
        }
        unsafe impl Send for SendPackage {}
    }

    pub fn free(mut self) -> (STREAM, PERIPHERY, BUF) {
        self.__await_completion();
        //Workaround for no-destructive patterns on Drop
        unsafe {
            let Self { stream, periphery, buffer, thread } = &self;
            let stream = ptr::read(stream as *const STREAM);
            let periphery = ptr::read(periphery as *const PERIPHERY);
            let buffer = ptr::read(buffer as *const BUF);
            let _thread = ptr::read(thread as *const _);
            mem::forget(self);
            (stream, periphery, buffer)
        }
    }

    // as we emulate it using a thread we can not "instant" cancel it
    // like in the hardware API
    fn __await_completion(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl<S,P,B> Drop for Transfer<S,P,B>
where
    S: Stream + Send,
    P: Periphery + Send,
    B: WriteBuffer<Word=u8>  + Send + 'static
{
    fn drop(&mut self) {
        self.__await_completion();
    }
}