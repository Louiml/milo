//! Version information for the `mpm` package manager.

/// The crate version, taken from Cargo.toml at build time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The package-manager protocol version. Bumped when the registry
/// protocol or the manifest format changes in an incompatible way.
pub const PROTOCOL_VERSION: &str = "1";
