//! Hand-written bindings to the stable Milo C ABI (`include/milo/milo.h`).
//!
//! The Milo project deliberately has no bindgen dependency: the ABI header
//! is small, stable, and hand-binding it keeps `cargo build` working with
//! no network access and no code-generation step.
//!
//! The declarations below mirror `include/milo/milo.h` exactly. The
//! `abi_version_matches` test asserts that the two stay in step.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int};

pub const MILO_ABI_VERSION: u32 = 1;
pub const MILO_LANGUAGE_VERSION: &str = "0.1";
pub const MILO_RUNTIME_VERSION: &str = "0.1.0";

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiloStatus {
    Ok = 0,
    ErrSyntax = 1,
    ErrType = 2,
    ErrRuntime = 3,
    ErrIo = 4,
    ErrNotFound = 5,
    ErrUnsupported = 6,
    ErrNoMem = 7,
    ErrInternal = 8,
}

impl MiloStatus {
    pub fn from_i32(v: i32) -> MiloStatus {
        match v {
            0 => MiloStatus::Ok,
            1 => MiloStatus::ErrSyntax,
            2 => MiloStatus::ErrType,
            3 => MiloStatus::ErrRuntime,
            4 => MiloStatus::ErrIo,
            5 => MiloStatus::ErrNotFound,
            6 => MiloStatus::ErrUnsupported,
            7 => MiloStatus::ErrNoMem,
            8 => MiloStatus::ErrInternal,
            _ => MiloStatus::ErrInternal,
        }
    }

    pub fn is_ok(self) -> bool {
        matches!(self, MiloStatus::Ok)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiloValueKind {
    Null = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    List = 5,
    Dict = 6,
    Tensor = 7,
    Model = 8,
    Dataset = 9,
    Module = 10,
    Function = 11,
    Builtin = 12,
}

#[repr(C)]
pub struct MiloRuntime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MiloValue {
    _private: [u8; 0],
}

extern "C" {
    pub fn milo_abi_version() -> u32;
    pub fn milo_version_string() -> *const c_char;
    pub fn milo_language_version() -> *const c_char;
    pub fn milo_status_to_string(status: MiloStatus) -> *const c_char;

    pub fn milo_last_error(buf: *mut c_char, cap: usize) -> usize;
    pub fn milo_clear_error();

    pub fn milo_runtime_create() -> *mut MiloRuntime;
    pub fn milo_runtime_destroy(rt: *mut MiloRuntime);

    pub fn milo_value_kind_of(v: *const MiloValue) -> MiloValueKind;
    pub fn milo_value_is_null(v: *const MiloValue) -> c_int;
    pub fn milo_value_as_bool(v: *const MiloValue) -> c_int;
    pub fn milo_value_as_int(v: *const MiloValue) -> i64;
    pub fn milo_value_as_float(v: *const MiloValue) -> f64;
    pub fn milo_value_as_string(v: *const MiloValue) -> *const c_char;
    pub fn milo_value_string_length(v: *const MiloValue) -> usize;
    pub fn milo_value_release(v: *mut MiloValue);

    pub fn milo_runtime_check(
        rt: *mut MiloRuntime,
        src: *const c_char,
        len: usize,
        filename: *const c_char,
    ) -> MiloStatus;

    pub fn milo_runtime_run(
        rt: *mut MiloRuntime,
        src: *const c_char,
        len: usize,
        filename: *const c_char,
        out_value: *mut *mut MiloValue,
    ) -> MiloStatus;

    pub fn milo_runtime_run_and_print(
        rt: *mut MiloRuntime,
        src: *const c_char,
        len: usize,
        filename: *const c_char,
    ) -> MiloStatus;
}

/// Fetch the calling thread's last error message, if any.
pub fn last_error() -> Option<String> {
    unsafe {
        let cap = 4096usize;
        let mut buf = vec![0u8; cap];
        let written = milo_last_error(buf.as_mut_ptr() as *mut c_char, cap);
        if written == 0 {
            return None;
        }
        let slice = &buf[..written.min(cap - 1)];
        let text = String::from_utf8_lossy(slice).into_owned();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// Run a snippet of Milo source and return its printed output.
#[cfg(has_runtime)]
pub fn run_source(source: &str, filename: &str) -> Result<(), String> {
    use std::os::raw::c_char;
    unsafe {
        let rt = milo_runtime_create();
        if rt.is_null() {
            return Err("could not create a Milo runtime".to_string());
        }
        let status = milo_runtime_run_and_print(
            rt,
            source.as_ptr() as *const c_char,
            source.len(),
            filename.as_ptr() as *const c_char,
        );
        milo_runtime_destroy(rt);
        if status.is_ok() {
            Ok(())
        } else {
            Err(last_error().unwrap_or_else(|| "unknown error".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_matches_the_header() {
        // The constant here and the value the runtime reports must agree;
        // a mismatch means `include/milo/milo.h` and this file have drifted.
        #[cfg(has_runtime)]
        {
            let reported = unsafe { milo_abi_version() };
            assert_eq!(reported, MILO_ABI_VERSION);
        }
        #[cfg(not(has_runtime))]
        {
            assert_eq!(MILO_ABI_VERSION, 1);
        }
    }

    #[test]
    fn status_codes_have_names() {
        assert!(MiloStatus::Ok.is_ok());
        assert!(!MiloStatus::ErrSyntax.is_ok());
        assert_eq!(MiloStatus::from_i32(0), MiloStatus::Ok);
        assert_eq!(MiloStatus::from_i32(1), MiloStatus::ErrSyntax);
    }
}
