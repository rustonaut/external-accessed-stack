use std::{cell::{RefCell, RefMut}, rc::Rc};

use rand::Rng;

use super::super::*;

pub type MockBufferMut<'r, 'a> = Pin<&'r mut RABufferAnchor<'a, u8, OpIntMock>>;

#[inline]
pub fn not(v: bool) -> bool {
    !v
}

/// This is very similar to how this should be used except
/// that we return a clone of the `op_int` and don't really
/// do anything with it (and normally you have a second buffer
/// or target address or similar).
pub async fn mock_operation<'r, 'a, T>(mut buffer: Pin<&'r mut RABufferAnchor<'a, T, OpIntMock>>) -> (OperationHandle<'a,'r, T, OpIntMock>, MockInfo) {
    buffer.as_mut().cancellation().await;
    let (buffer_start, len) = buffer.as_mut().get_buffer_ptr_and_len();
    let (op_int, start, mock_info) = OpIntMock::new(buffer_start, len);
    // Safe: We pass in the right opt_int (well as we don't actually access the buffer it kinda doesn't matter)
    let op_handle = unsafe { buffer.try_register_new_operation(op_int) };
    // Unwrap Safe: We awaited completion
    let op_handle = op_handle.unwrap();
    // we don't do really start anything here but if you do implement a real operation
    // you would need to start the operation around here (after you got the op_handle).
    start();
    (op_handle, mock_info)
}

//TODO also track dropping
pub struct OpIntMock {
    pub mock_info: Rc<RefCell<CallInfo>>,
}

impl OpIntMock {

    pub fn new<T>(_buffer_start: *mut T,  _len: usize) -> (Self, impl FnOnce(), MockInfo) {
        let mock_info: Rc<RefCell<_>> = Default::default();
        let op_int = OpIntMock { mock_info: mock_info.clone()  };
        let start_op = || {};
        (op_int, start_op, MockInfo { mock_info })
    }

    fn fixed_address_mock_call(&mut self) -> RefMut<CallInfo> {
        let addr = self as *mut _;
        let mut mock_info = self.mock_info.borrow_mut();
        mock_info.op_int_addr.push(addr);
        mock_info
    }
}

impl Drop for OpIntMock {

    fn drop(&mut self)  {
        let mut mock_info = self.fixed_address_mock_call();
        mock_info.was_dropped = true;
    }
}

unsafe impl OperationInteraction for OpIntMock {
    fn make_sure_operation_ended(mut self: Pin<&mut Self>) {
        let mut mock_info = self.fixed_address_mock_call();
        mock_info.called_make_sure_operation_ended = true;
    }

    fn poll_completion(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> {
        let mut mock_info = self.fixed_address_mock_call();
        mock_info.called_poll_completion = true;
        if mock_info.yields_before_completion_ready > 0 {
            mock_info.yields_before_completion_ready -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }

    fn poll_cancel(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> {
        let mut mock_info = self.fixed_address_mock_call();
        if mock_info.called_poll_completion {
            return Poll::Ready(());
        }
        mock_info.called_poll_cancel = true;
        if mock_info.yields_before_cancel_ready > 0 {
            mock_info.yields_before_cancel_ready -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

#[derive(Clone)]
pub struct MockInfo {
    pub mock_info: Rc<RefCell<CallInfo>>,
}


pub struct CallInfo {
    pub called_make_sure_operation_ended: bool,
    pub called_poll_completion: bool,
    pub yields_before_completion_ready: usize,
    pub called_poll_cancel: bool,
    pub yields_before_cancel_ready: usize,
    pub op_int_addr: Vec<*mut OpIntMock>,
    pub was_dropped: bool,
}

impl Default for CallInfo {
    fn default() -> Self {
        //FIXME proptest?
        let mut rng = rand::thread_rng();
        CallInfo {
            called_make_sure_operation_ended: false,
            called_poll_cancel: false,
            called_poll_completion: false,
            yields_before_cancel_ready: rng.gen_range(0,5),
            yields_before_completion_ready: rng.gen_range(0,5),
            was_dropped: false,
            op_int_addr: Vec::new(),
        }
    }
}


impl MockInfo {

    pub fn assert_was_dropped(&self) {
        let mock_info = self.mock_info.borrow();
        assert!(mock_info.was_dropped);
    }

    pub fn assert_not_run(&self) {
        let mock_info = self.mock_info.borrow();
        assert_eq!(mock_info.called_make_sure_operation_ended, false);
        assert_eq!(mock_info.called_poll_cancel, false);
        assert_eq!(mock_info.called_poll_completion, false);
    }

    pub fn assert_completion_run(&self) {
        let mock_info = self.mock_info.borrow();
        assert_eq!(mock_info.called_make_sure_operation_ended, true);
        assert_eq!(mock_info.called_poll_cancel, false);
        assert_eq!(mock_info.called_poll_completion, true);
    }

    pub fn assert_cancellation_run(&self) {
        let mock_info = self.mock_info.borrow();
        assert_eq!(mock_info.called_make_sure_operation_ended, true);
        assert_eq!(mock_info.called_poll_cancel, true);
        assert_eq!(mock_info.called_poll_completion, true);
    }

    pub fn assert_notify_cancel_run(&self) {
        let mock_info = self.mock_info.borrow();
        assert_eq!(mock_info.called_make_sure_operation_ended, false);
        assert_eq!(mock_info.called_poll_cancel, true);
        assert_eq!(mock_info.called_poll_completion, false);

    }

    pub fn assert_op_ended_enforced(&self) {
        let mock_info = self.mock_info.borrow();
        assert_eq!(mock_info.called_make_sure_operation_ended, true);
    }

    pub fn assert_op_int_addr_eq(&self, addr: *mut OpIntMock) {
        let mock_info = self.mock_info.borrow();
        if mock_info.op_int_addr.is_empty() {
            panic!("No operation on op int was called, can't do addr eq.");
        }
        assert!(mock_info.op_int_addr.iter().all(|call_addr| *call_addr == addr));
    }
}
