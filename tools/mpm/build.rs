//! Locates the Milo C runtime static library so that `mpm` can link it.
//!
//! The C++ runtime is built by CMake, not Cargo. `build.rs` searches the
//! usual locations for `milo_abi` and, if it finds it, emits the link
//! directives and sets `cfg(has_runtime)`. If it does not find it, the
//! crate still builds — the runtime is only needed for the commands that
//! actually execute Milo code.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MILO_RUNTIME_LIB_DIR");
    println!("cargo:rustc-check-cfg=cfg(has_runtime)");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir.clone());

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("MILO_RUNTIME_LIB_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    for preset in ["debug", "release", "relwithdebinfo"] {
        candidates.push(repo_root.join("build").join(preset).join("lib"));
    }

    for dir in &candidates {
        if let Some(lib) = find_library(dir) {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=dylib=milo");
            println!("cargo:rustc-cfg=has_runtime");
            println!("cargo:rerun-if-changed={}", lib.display());
            stage_runtime(dir);
            return;
        }
    }

    println!(
        "cargo:warning=Milo C runtime not found; `mpm` will build without runtime support. \
         Build it with `python build.py` first."
    );
}

/// Find the Milo shared library in `dir`.
///
/// The library is called `milo.dll` on Windows (with an import library
/// beside it) and `libmilo.so` on other platforms.
fn find_library(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["milo.lib", "milo.dll.lib", "milo_shared.lib"]
    } else {
        &["libmilo.so", "libmilo.dylib"]
    };
    for name in names {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Make the shared library findable at run time.
///
/// On Windows the DLL has to sit next to the executable, so it is copied
/// into Cargo's target directory. On other platforms the library's own
/// directory is added to the runtime search path.
fn stage_runtime(lib_dir: &Path) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
    // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.clone());

    if cfg!(target_os = "windows") {
        // The DLL lives in ../bin next to the import library.
        let dll = lib_dir
            .parent()
            .map(|p| p.join("bin").join("milo.dll"))
            .unwrap_or_default();
        if dll.is_file() {
            let _ = std::fs::copy(&dll, target_dir.join("milo.dll"));
            // Test binaries live one directory deeper.
            let deps = target_dir.join("deps");
            if deps.is_dir() {
                let _ = std::fs::copy(&dll, deps.join("milo.dll"));
            }
            println!("cargo:rerun-if-changed={}", dll.display());
        }
    } else {
        // Record the directory so the loader finds the library.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}
