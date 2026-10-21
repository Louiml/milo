//! `mpm` — the Milo package manager.
//!
//! This crate is the Rust side of the project. It talks to the Milo runtime
//! exclusively through the stable C ABI declared in `include/milo/milo.h`;
//! it never links a C++ symbol directly.

pub mod cache;
pub mod client;
pub mod ffi;
pub mod json;
pub mod lockfile;
pub mod manifest;
pub mod package;
pub mod registry;
pub mod resolver;
pub mod semver;
pub mod sha256;
pub mod version;

/// The canonical name of the manifest file.
pub const MANIFEST_FILE: &str = "milo.toml";

/// The canonical name of the lockfile.
pub const LOCK_FILE: &str = "milo.lock";

/// Print the `mpm` version banner.
pub fn version_line() -> String {
    format!("mpm {}", version::VERSION)
}
