//! The end-to-end tests for `mpm`.
//!
//! They run the real binary, in a temporary directory, through the whole
//! story: a package is written, built, published to a local registry,
//! found by another project, installed, and installed again with the
//! network turned off.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The `mpm` binary that Cargo just built.
fn mpm() -> &'static str {
    env!("CARGO_BIN_EXE_mpm")
}

/// A directory of its own for one test.
fn scratch(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("mpm-test-{name}"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// Run `mpm` in a directory.
fn run(directory: &Path, arguments: &[&str]) -> Output {
    Command::new(mpm())
        .args(arguments)
        .current_dir(directory)
        .output()
        .expect("mpm could not be run")
}

/// Run `mpm` and return what it printed.
fn output(directory: &Path, arguments: &[&str]) -> String {
    let result = run(directory, arguments);
    assert!(
        result.status.success(),
        "mpm {arguments:?} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8_lossy(&result.stdout).into_owned()
}

#[test]
fn the_whole_story_works() {
    let root = scratch("story");
    let registry = root.join("registry");

    // A library, built and published.
    let library = root.join("milo-lib");
    std::fs::create_dir_all(&library).unwrap();
    output(&library, &["init", "milo-lib"]);
    assert!(library.join("milo.toml").is_file());
    output(&library, &["build"]);
    let built = library.join("target").join("milo-lib-0.1.0.mpkg");
    assert!(built.is_file(), "the package was not built");

    let published = output(&library, &["publish", "--registry", registry.to_str().unwrap()]);
    assert!(published.contains("published milo-lib 0.1.0"), "{published}");
    assert!(registry.join("index.json").is_file());
    assert!(registry.join("milo-lib-0.1.0.mpkg").is_file());

    // The registry knows about it.
    let found = output(&root, &["search", "--registry", registry.to_str().unwrap(), "milo-lib"]);
    assert!(found.contains("milo-lib"), "{found}");
    assert!(found.contains("0.1.0"), "{found}");

    // Another project depends on it.
    let app = root.join("app");
    std::fs::create_dir_all(&app).unwrap();
    output(&app, &["init", "my-app"]);
    output(&app, &["add", "milo-lib@^0.1"]);
    let installed = output(
        &app,
        &["install", "--registry", registry.to_str().unwrap()],
    );
    assert!(installed.contains("installed milo-lib 0.1.0"), "{installed}");
    assert!(
        app.join("milo_packages").join("milo-lib").join("milo.toml").is_file(),
        "the dependency was not unpacked"
    );

    // The lockfile says exactly what was installed.
    let lockfile = std::fs::read_to_string(app.join("milo.lock")).unwrap();
    assert!(lockfile.contains("milo-lib"), "{lockfile}");
    assert!(lockfile.contains("0.1.0"), "{lockfile}");
    assert!(lockfile.contains("sha256"), "{lockfile}");

    // And the same install works with the network turned off, because
    // everything is in the cache by now.
    let offline = output(&app, &["install", "--offline"]);
    assert!(offline.contains("installed milo-lib 0.1.0"), "{offline}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_unknown_package_is_reported() {
    let root = scratch("unknown");
    let registry = root.join("registry");
    std::fs::create_dir_all(&registry).unwrap();
    std::fs::write(
        registry.join("index.json"),
        r#"{"protocol": 1, "packages": {}}"#,
    )
    .unwrap();

    let app = root.join("app");
    std::fs::create_dir_all(&app).unwrap();
    output(&app, &["init", "my-app"]);
    output(&app, &["add", "not-there"]);

    let result = run(
        &app,
        &["install", "--registry", registry.to_str().unwrap()],
    );
    assert!(!result.status.success());
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(message.contains("not-there"), "{message}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_package_that_does_not_match_its_hash_is_refused() {
    let root = scratch("hash");
    let registry = root.join("registry");
    std::fs::create_dir_all(&registry).unwrap();

    // A registry whose index points at a file that is not what it says.
    std::fs::write(registry.join("milo-lib-0.1.0.mpkg"), b"not a package").unwrap();
    std::fs::write(
        registry.join("index.json"),
        r#"{
            "protocol": 1,
            "packages": {
                "milo-lib": {
                    "versions": {
                        "0.1.0": {
                            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                            "size": 11,
                            "url": "milo-lib-0.1.0.mpkg"
                        }
                    }
                }
            }
        }"#,
    )
    .unwrap();

    let app = root.join("app");
    std::fs::create_dir_all(&app).unwrap();
    output(&app, &["init", "my-app"]);
    output(&app, &["add", "milo-lib"]);

    let result = run(
        &app,
        &["install", "--registry", registry.to_str().unwrap()],
    );
    assert!(!result.status.success());
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(message.contains("hash"), "{message}");

    let _ = std::fs::remove_dir_all(&root);
}
