use core::{
    marker::PhantomData,
    mem,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Trait to set default functions, workaround to not being able to have fn-ptr's in const fns
pub trait AtomicHookSettings {
    type Param;
    type Ret;

    fn default_hook() -> fn(Self::Param) -> Self::Ret;
}

#[macro_export]
macro_rules! define_atomic_hooks {
    ($(
        $(#[$attr:meta])*
        $v:vis static $name:ident: AtomicHook<$dname:ident : fn($param:ty) -> $ret:ty > = $default:ident;
    )*) => ($(
        $v struct $dname;
        impl $crate::utils::AtomicHookSettings for $dname {
            type Param = $param;
            type Ret = $ret;

            fn default_hook() -> fn($param) -> $ret {
                $default
            }
        }
        $(#[$attr])*
        $v static $name: $crate::utils::AtomicHook<$dname> = $crate::utils::AtomicHook {
            ptr: core::sync::atomic::AtomicUsize::new(0),
            marker: core::marker::PhantomData
        };
    )*);
}

/// A function hook based on `AtomicPtr`.
///
/// This only works with functions not closures.
///
/// Due to not having variable number of generic arguments the function
/// always has one parameter, as such use a tuple to have any other number
/// of parameters:
///
/// - `fn()` => `fn(())` => `AtomicHook<(),()>`
/// - `fn(A) -> B` => `AtomicHook<A,B>`
/// - `fn(A,B) -> C` => `fn((A,B)) -> C` => `AtomicHook<(A,B), C>`
pub struct AtomicHook<Settings>
where
    Settings: AtomicHookSettings,
{
    //FIXME: Check if there is a single architecture with a function pointer size which
    //       is not the normal pointer size and which supports C99. We have no guarantee
    //       from the standard that storing the pointer in a usize is save but I really
    //       couldn't (fastly) find any platform where this is a problem. (C99 just specifies
    //       that all function pointers have the same size, but not that it's the same size as
    //       other pointers)
    //
    //       Note: that we use transmute to turn the usize back into a pointer which fails to
    //       compile if the size doesn't match so this won't cause runtime problems and only
    //       will fail to compile on targets which have different function pointer sizes then
    //       they have data pointer sizes.
    #[doc(hidden)]
    pub(crate) ptr: AtomicUsize,
    #[doc(hidden)]
    pub(crate) marker: PhantomData<fn(Settings::Param) -> Settings::Ret>,
}

impl<Settings> AtomicHook<Settings>
where
    Settings: AtomicHookSettings,
{
    fn fn_to_usize(func: fn(Settings::Param) -> Settings::Ret) -> usize {
        func as usize
    }

    /// Converts back to an function pointer.
    ///
    /// # Unsafe-Contract
    ///
    /// This must only be called with values retrieved from the private
    /// `ptr` field. I.e. only with usize values which are either 0 or
    /// represent a valid function pointer address
    unsafe fn usize_to_fn(func: usize) -> fn(Settings::Param) -> Settings::Ret {
        if func == 0 {
            Settings::default_hook()
        } else {
            // This will fail to compile if the size of usize and the fn ptr doesn't match.
            //SAFE: We know it's either 0 or a valid func ptr address.
            //      As we already checked for 0 this is a valid func ptr address.
            mem::transmute(func)
        }
    }

    /// Returns a function pointer to the hook
    pub fn get(&self) -> fn(Settings::Param) -> Settings::Ret {
        let fnptr_as_usize = self.ptr.load(Ordering::Acquire);
        //SAFE: Requires to be called with values from self.ptr only.
        unsafe { Self::usize_to_fn(fnptr_as_usize) }
    }

    /// Replace given atomic hook
    pub fn replace(
        &self,
        new_hook: fn(Settings::Param) -> Settings::Ret,
    ) -> fn(Settings::Param) -> Settings::Ret {
        let fnptr_as_usize = self.ptr.swap(Self::fn_to_usize(new_hook), Ordering::AcqRel);
        //SAFE: Requires to be called with values from self.ptr only.
        unsafe { Self::usize_to_fn(fnptr_as_usize) }
    }
}

#[cfg(test)]
mod tests {
    use core::mem;

    fn default_test_fn(val: u8) -> u8 {
        val + 18
    }

    #[test]
    fn is_initialized_with_the_right_function() {
        define_atomic_hooks! {
            static TEST_HOOK: AtomicHook<DefaultTestFn: fn(u8) -> u8> = default_test_fn;
        }

        let func = TEST_HOOK.get();
        assert_eq!(func(7), 25);
        assert_eq!(func as usize, default_test_fn as usize);
    }

    #[test]
    fn the_function_can_be_replaced() {
        define_atomic_hooks! {
            static TEST_HOOK: AtomicHook<DefaultTestFn: fn(u8) -> u8> = default_test_fn;
        }

        fn replace_fn(v: u8) -> u8 {
            v + 25
        }
        fn replace_fn2(v: u8) -> u8 {
            v + 32
        }
        let old_fn = TEST_HOOK.replace(replace_fn);
        let current = TEST_HOOK.get();
        assert_eq!(old_fn(7), 25);
        assert_eq!(current(7), 32);
        assert_eq!(current as usize, replace_fn as usize);
        assert_eq!(old_fn as usize, default_test_fn as usize);

        let old_fn = TEST_HOOK.replace(replace_fn2);
        let current = TEST_HOOK.get();
        assert_eq!(old_fn(7), 32);
        assert_eq!(current(7), 39);
        assert_eq!(current as usize, replace_fn2 as usize);
        assert_eq!(old_fn as usize, replace_fn as usize);
    }

    #[test]
    fn size_of_fn_pointers_is_as_expected() {
        assert_eq!(mem::size_of::<fn()>(), mem::size_of::<usize>());
    }
}
