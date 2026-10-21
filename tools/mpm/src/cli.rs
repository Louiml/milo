//! The `mpm` command-line interface.
//!
//! The command set is fixed early so that scripts and CI can rely on it:
//!
//! ```text
//!   mpm init [name]        create a milo.toml manifest in the current directory
//!   mpm add <spec>         add a dependency to the manifest
//!   mpm remove <name>      remove a dependency
//!   mpm install            resolve and install the dependency tree
//!   mpm build              build the current package
//!   mpm publish            publish the current package to the registry
//!   mpm update [name]      update dependencies to the latest allowed versions
//!   mpm search <query>     search the registry index
//! ```
//!
//! Every command that touches the network says so, and `--offline` turns
//! the network off for good: everything has to come from the cache.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mpm::cache::Cache;
use mpm::client::Registry;
use mpm::json;
use mpm::lockfile::{Lockfile, LockedPackage};
use mpm::manifest;
use mpm::package::Package;
use mpm::registry::PackageSpec;
use mpm::resolver;
use mpm::semver::{Requirement, Version};

/// The directory that installed packages are unpacked into.
const VENDOR_DIRECTORY: &str = "milo_packages";

/// The environment variable that names the registry.
const REGISTRY_VARIABLE: &str = "MILO_REGISTRY";

const USAGE: &str = "\n\
mpm - the Milo package manager

Usage:
  mpm <command> [options]

Commands:
  init [name]        Create a milo.toml manifest in the current directory
  add <spec>         Add a dependency to the manifest
  remove <name>      Remove a dependency from the manifest
  install            Resolve and install the dependency tree
  build              Build the current package
  publish            Publish the current package to the registry
  update [name]      Update dependencies to the latest allowed versions
  search <query>     Search the registry index
  runtime            Report the runtime this build is linked against
  help               Show this message
  version            Show version information

Options:
  --offline          Never touch the network; use only the local cache
  --manifest <path>  Use a specific manifest file
  --registry <url>   Use a specific registry index

Examples:
  mpm init my-project
  mpm add \"milo-numpy@^0.2\"
  mpm install --offline
";

pub fn run(args: Vec<String>) -> ExitCode {
    let mut positional: Vec<String> = Vec::new();
    let mut flags: Vec<String> = Vec::new();

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if arg.starts_with("--") {
            // Options that take a value consume the next argument.
            if matches!(arg.as_str(), "--manifest" | "--registry") {
                flags.push(arg);
                if let Some(value) = iter.next() {
                    flags.push(value);
                }
            } else {
                flags.push(arg);
            }
        } else {
            positional.push(arg);
        }
    }

    let Some(command) = positional.first().cloned() else {
        print!("{USAGE}");
        return ExitCode::from(2);
    };

    let rest: Vec<String> = positional[1..].to_vec();
    let offline = flags.iter().any(|f| f == "--offline");
    let manifest_file = flag_value(&flags, "--manifest")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(mpm::MANIFEST_FILE));
    let registry = flag_value(&flags, "--registry")
        .or_else(|| std::env::var(REGISTRY_VARIABLE).ok())
        .unwrap_or_else(|| ".milo/registry".to_string());

    match command.as_str() {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("{}", mpm::version_line());
            println!("registry protocol: {}", mpm::version::PROTOCOL_VERSION);
            println!("language version:   {}", mpm::ffi::MILO_LANGUAGE_VERSION);
            ExitCode::SUCCESS
        }
        "init" => cmd_init(&rest),
        "add" => cmd_add(&rest),
        "remove" => cmd_remove(&rest),
        "runtime" => cmd_runtime(),
        "install" => cmd_install(&manifest_file, &registry, offline),
        "build" => cmd_build(&manifest_file),
        "publish" => cmd_publish(&manifest_file, &registry),
        "update" => cmd_update(&manifest_file, &registry, offline, rest.first()),
        "search" => cmd_search(&registry, rest.first().map(String::as_str)),
        other => {
            eprintln!("error: unknown command `{other}`");
            eprintln!("try `mpm help`");
            ExitCode::from(2)
        }
    }
}

/// The value of an option, if it was given.
fn flag_value(flags: &[String], name: &str) -> Option<String> {
    let index = flags.iter().position(|f| f == name)?;
    flags.get(index + 1).cloned()
}

fn manifest_path() -> PathBuf {
    PathBuf::from(mpm::MANIFEST_FILE)
}

/// The name and version of the package in a manifest.
fn package_identity(path: &Path) -> Result<(String, Version), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table = manifest::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let package = table
        .get("package")
        .ok_or_else(|| format!("{}: there is no [package] table", path.display()))?;
    let name = package
        .as_table()
        .and_then(|t| t.get("name"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: the package has no name", path.display()))?
        .to_string();
    let version_text = package
        .as_table()
        .and_then(|t| t.get("version"))
        .and_then(|v| v.as_str())
        .unwrap_or("0.1.0");
    let version = Version::parse(version_text)?;
    Ok((name, version))
}

/// The dependencies of a manifest, as requirements.
fn manifest_requirements(path: &Path) -> Result<Vec<(String, Requirement)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let table = manifest::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut requirements = Vec::new();
    if let Some(manifest::Value::Table(dependencies)) = table.get("dependencies") {
        for (name, value) in dependencies {
            let text = value.as_str().unwrap_or("*");
            requirements.push((name.clone(), Requirement::parse(text)?));
        }
    }
    requirements.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(requirements)
}

fn cmd_init(args: &[String]) -> ExitCode {
    let name = args.first().cloned().unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "milo-package".to_string())
    });

    if !is_valid_package_name(&name) {
        eprintln!("error: `{name}` is not a valid package name");
        eprintln!("  names must start with a letter and contain only letters,");
        eprintln!("  digits, `-` and `_`");
        return ExitCode::from(2);
    }

    let path = manifest_path();
    if path.exists() {
        eprintln!("error: {} already exists", path.display());
        return ExitCode::from(1);
    }

    let contents = format!(
        "[package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2026\"\n\
         description = \"\"\n\
         license = \"MIT\"\n\
         \n\
         [dependencies]\n"
    );

    if let Err(e) = fs::write(&path, contents) {
        eprintln!("error: could not write {}: {e}", path.display());
        return ExitCode::from(1);
    }

    println!("created {}", path.display());
    println!("next: add dependencies with `mpm add <name>`");
    ExitCode::SUCCESS
}

fn cmd_add(args: &[String]) -> ExitCode {
    let Some(spec_text) = args.first() else {
        eprintln!("error: `mpm add` needs a package specification");
        eprintln!("  example: mpm add \"milo-numpy@^0.2\"");
        return ExitCode::from(2);
    };

    let spec = match PackageSpec::parse(spec_text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let path = manifest_path();
    if !path.exists() {
        eprintln!("error: no {} found; run `mpm init` first", path.display());
        return ExitCode::from(1);
    }

    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: could not read {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    let mut table = match manifest::parse(&text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {} is not valid TOML: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    let deps = table
        .entry("dependencies".to_string())
        .or_insert_with(|| manifest::Value::Table(Default::default()));
    let manifest::Value::Table(deps) = deps else {
        eprintln!("error: `dependencies` in {} is not a table", path.display());
        return ExitCode::from(1);
    };

    if deps.contains_key(&spec.name) {
        eprintln!("error: `{}` is already a dependency", spec.name);
        return ExitCode::from(1);
    }

    deps.insert(
        spec.name.clone(),
        manifest::Value::Str(spec.requirement.to_string()),
    );

    let rendered = render_manifest(&table);
    if let Err(e) = fs::write(&path, rendered) {
        eprintln!("error: could not write {}: {e}", path.display());
        return ExitCode::from(1);
    }

    println!("added {} {}", spec.name, spec.requirement);
    ExitCode::SUCCESS
}

fn cmd_remove(args: &[String]) -> ExitCode {
    let Some(name) = args.first() else {
        eprintln!("error: `mpm remove` needs a package name");
        return ExitCode::from(2);
    };

    let path = manifest_path();
    if !path.exists() {
        eprintln!("error: no {} found", path.display());
        return ExitCode::from(1);
    }

    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: could not read {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    let mut table = match manifest::parse(&text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {} is not valid TOML: {e}", path.display());
            return ExitCode::from(1);
        }
    };

    let removed = match table.get_mut("dependencies") {
        Some(manifest::Value::Table(deps)) => deps.remove(name).is_some(),
        _ => false,
    };

    if !removed {
        eprintln!("error: `{name}` is not a dependency");
        return ExitCode::from(1);
    }

    let rendered = render_manifest(&table);
    if let Err(e) = fs::write(&path, rendered) {
        eprintln!("error: could not write {}: {e}", path.display());
        return ExitCode::from(1);
    }

    println!("removed {name}");
    ExitCode::SUCCESS
}

/// Report the runtime that `mpm` is linked against.
///
/// This is how you check that the C ABI bridge is working: it calls the
/// runtime through `include/milo/milo.h` and nothing else.
fn cmd_runtime() -> ExitCode {
    #[cfg(has_runtime)]
    {
        use mpm::ffi;
        use std::ffi::CStr;

        let runtime_version = unsafe { CStr::from_ptr(ffi::milo_version_string()) };
        let language_version = unsafe { CStr::from_ptr(ffi::milo_language_version()) };
        let abi = unsafe { ffi::milo_abi_version() };

        println!("runtime:   {}", runtime_version.to_string_lossy());
        println!("language:  {}", language_version.to_string_lossy());
        println!("abi:       {abi}");

        if abi != ffi::MILO_ABI_VERSION {
            eprintln!("error: the runtime's ABI version ({abi}) does not match the one this");
            eprintln!("       build expects ({}). Rebuild the runtime, or use a matching", ffi::MILO_ABI_VERSION);
            eprintln!("       version of `mpm`.");
            return ExitCode::from(1);
        }
        ExitCode::SUCCESS
    }

    #[cfg(not(has_runtime))]
    {
        eprintln!("error: this build of `mpm` is not linked against the Milo runtime");
        eprintln!("  build the C++ runtime first: python build.py");
        ExitCode::from(1)
    }
}

/// The directory a manifest lives in.
///
/// A manifest called `milo.toml` has no parent directory as far as the
/// file system is concerned, so the current one is used.
fn package_directory(manifest_file: &Path) -> PathBuf {
    manifest_file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

/// `mpm install` — resolve and install the dependency tree.
fn cmd_install(manifest_file: &Path, registry: &str, offline: bool) -> ExitCode {
    if !manifest_file.is_file() {
        eprintln!("error: no {} found; run `mpm init` first", manifest_file.display());
        return ExitCode::from(1);
    }
    let requirements = match manifest_requirements(manifest_file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let lock_path = manifest_file
        .parent()
        .unwrap_or(Path::new("."))
        .join(mpm::lockfile::FILE);

    // A lockfile that still satisfies the manifest is the answer; there
    // is no reason to go looking for a newer one.
    let locked = Lockfile::read(&lock_path).ok();
    let mut use_lock = false;
    if let Some(lock) = &locked {
        use_lock = requirements.iter().all(|(name, requirement)| {
            match lock.find(name) {
                Some(package) => requirement.matches(&package.version),
                None => false,
            }
        });
    }

    let mut packages = Vec::new();
    if use_lock {
        let lock = locked.as_ref().unwrap();
        for package in &lock.packages {
            packages.push(resolver::Resolved {
                name: package.name.clone(),
                version: package.version.clone(),
                sha256: package.sha256.clone(),
                url: package.source.clone(),
            });
        }
    } else {
        let index = match fetch_index(registry, offline) {
            Ok(index) => index,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        };
        let resolved = match resolver::resolve(&requirements, &index) {
            Ok(resolved) => resolved,
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("  the requirements in {} cannot be satisfied", manifest_file.display());
                return ExitCode::from(1);
            }
        };
        packages = resolved;
    }

    let cache = match Cache::default_location() {
        Ok(cache) => cache,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let vendor = PathBuf::from(VENDOR_DIRECTORY);
    let mut installed = Vec::new();
    for package in &packages {
        let bytes = match fetch_package(&cache, package, registry, offline) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        };
        let archive = match Package::from_bytes(&bytes) {
            Ok(archive) => archive,
            Err(e) => {
                eprintln!("error: {} {}: {e}", package.name, package.version);
                return ExitCode::from(1);
            }
        };
        let directory = vendor.join(&package.name);
        let _ = fs::remove_dir_all(&directory);
        if let Err(e) = archive.unpack(&directory) {
            eprintln!("error: could not unpack {}: {e}", package.name);
            return ExitCode::from(1);
        }
        println!("installed {} {}", package.name, package.version);
        installed.push(LockedPackage {
            name: package.name.clone(),
            version: package.version.clone(),
            sha256: package.sha256.clone(),
            source: package.url.clone(),
        });
    }

    let lockfile = Lockfile {
        version: 1,
        packages: installed,
    };
    if let Err(e) = lockfile.write(&lock_path) {
        eprintln!("error: could not write {}: {e}", lock_path.display());
        return ExitCode::from(1);
    }
    println!("wrote {}", lock_path.display());
    ExitCode::SUCCESS
}

/// The registry index, or an explanation of why there is none.
fn fetch_index(registry: &str, offline: bool) -> Result<mpm::registry::Index, String> {
    if offline {
        return Err(
            "the index is not in the cache, and --offline was asked for".to_string(),
        );
    }
    Registry::new(registry).index()
}

/// The bytes of a package, from the cache or from the registry.
fn fetch_package(
    cache: &Cache,
    package: &resolver::Resolved,
    registry: &str,
    offline: bool,
) -> Result<Vec<u8>, String> {
    if cache.contains(&package.sha256) {
        let bytes = std::fs::read(cache.path_for_hash(&package.sha256))
            .map_err(|e| format!("the cached package could not be read: {e}"))?;
        mpm::sha256::verify(&bytes, &package.sha256)?;
        return Ok(bytes);
    }
    if offline {
        return Err(format!(
            "{} {} is not in the cache, and --offline was asked for",
            package.name, package.version
        ));
    }
    // A relative URL is resolved against the registry.
    let url = if package.url.contains("://") {
        package.url.clone()
    } else {
        let base = registry.trim_end_matches('/');
        format!("{base}/{}", package.url)
    };
    let bytes = Registry::new(registry).download(&url)?;
    mpm::sha256::verify(&bytes, &package.sha256)?;
    // Put it in the cache, so that the next install does not need the
    // network.
    if let Ok(archive) = Package::from_bytes(&bytes) {
        let _ = cache.store(&archive);
    }
    Ok(bytes)
}

/// `mpm build` — pack the current package.
fn cmd_build(manifest_file: &Path) -> ExitCode {
    if !manifest_file.is_file() {
        eprintln!("error: no {} found; run `mpm init` first", manifest_file.display());
        return ExitCode::from(1);
    }
    let (name, version) = match package_identity(manifest_file) {
        Ok(identity) => identity,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let directory = package_directory(manifest_file);
    let package = match Package::from_directory(&name, &version, &directory) {
        Ok(package) => package,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let bytes = package.to_bytes();
    let hash = mpm::sha256::digest(&bytes).to_hex();
    let target = PathBuf::from("target").join(format!("{name}-{version}.mpkg"));
    if let Some(parent) = target.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("error: could not create {}: {e}", parent.display());
            return ExitCode::from(1);
        }
    }
    if let Err(e) = fs::write(&target, &bytes) {
        eprintln!("error: could not write {}: {e}", target.display());
        return ExitCode::from(1);
    }
    println!("built {}", target.display());
    println!("sha256 {hash}");
    ExitCode::SUCCESS
}

/// `mpm publish` — put the current package into a registry.
fn cmd_publish(manifest_file: &Path, registry: &str) -> ExitCode {
    if !manifest_file.is_file() {
        eprintln!("error: no {} found; run `mpm init` first", manifest_file.display());
        return ExitCode::from(1);
    }
    let (name, version) = match package_identity(manifest_file) {
        Ok(identity) => identity,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let directory = package_directory(manifest_file);
    let package = match Package::from_directory(&name, &version, &directory) {
        Ok(package) => package,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let bytes = package.to_bytes();
    let hash = mpm::sha256::digest(&bytes).to_hex();

    // Only a local registry can be published to: publishing is a write,
    // and the protocol has no write API.
    let registry_path = PathBuf::from(registry.trim_start_matches("file://"));
    if registry_path.starts_with("http://") || registry_path.starts_with("https://") {
        eprintln!("error: `{registry}` is not a local registry directory");
        eprintln!("  a registry that is served over HTTP cannot be written to");
        return ExitCode::from(1);
    }
    if !registry_path.is_dir() {
        if let Err(e) = fs::create_dir_all(&registry_path) {
            eprintln!("error: could not create {}: {e}", registry_path.display());
            return ExitCode::from(1);
        }
    }
    let file_name = format!("{name}-{version}.{}", mpm::package::EXTENSION);
    let archive_path = registry_path.join(&file_name);
    if let Err(e) = fs::write(&archive_path, &bytes) {
        eprintln!("error: could not write {}: {e}", archive_path.display());
        return ExitCode::from(1);
    }

    // Add the release to the index.
    let index_path = registry_path.join(mpm::client::INDEX_FILE);
    let mut index = match fs::read_to_string(&index_path) {
        Ok(text) => match json::parse(&text) {
            Ok(value) => match mpm::registry::Index::from_json(&value) {
                Ok(index) => index,
                Err(e) => {
                    eprintln!("error: {index_path:?}: {e}");
                    return ExitCode::from(1);
                }
            },
            Err(e) => {
                eprintln!("error: {index_path:?}: {e}");
                return ExitCode::from(1);
            }
        },
        Err(_) => mpm::registry::Index {
            protocol: mpm::registry::PROTOCOL_VERSION,
            packages: Vec::new(),
        },
    };

    let release = mpm::registry::PackageRelease {
        version: version.clone(),
        sha256: hash.clone(),
        size: bytes.len() as u64,
        url: file_name.clone(),
        signature: None,
        // The index carries the dependencies, so that a resolver can walk
        // the tree without reading the package.
        dependencies: manifest_requirements(manifest_file).unwrap_or_default(),
    };
    match index.packages.iter_mut().find(|p| p.name == name) {
        Some(entry) => {
            entry.releases.retain(|r| r.version != version);
            entry.releases.push(release);
        }
        None => index.packages.push(mpm::registry::PackageEntry {
            name: name.clone(),
            releases: vec![release],
        }),
    }
    let text = json::to_string_pretty(&index.to_json());
    if let Err(e) = fs::write(&index_path, text) {
        eprintln!("error: could not write {}: {e}", index_path.display());
        return ExitCode::from(1);
    }
    println!("published {} {} to {}", name, version, registry);
    println!("sha256 {hash}");
    ExitCode::SUCCESS
}

/// `mpm update` — resolve again, ignoring the lockfile.
fn cmd_update(
    manifest_file: &Path,
    registry: &str,
    offline: bool,
    only: Option<&String>,
) -> ExitCode {
    if !manifest_file.is_file() {
        eprintln!("error: no {} found; run `mpm init` first", manifest_file.display());
        return ExitCode::from(1);
    }
    let mut requirements = match manifest_requirements(manifest_file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(name) = only {
        if !requirements.iter().any(|(n, _)| n == name) {
            eprintln!("error: `{name}` is not a dependency");
            return ExitCode::from(1);
        }
        requirements.retain(|(n, _)| n == name);
    }
    let index = match fetch_index(registry, offline) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let resolved = match resolver::resolve(&requirements, &index) {
        Ok(resolved) => resolved,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    for package in resolved {
        println!("{} {}", package.name, package.version);
    }
    println!("run `mpm install` to install the new versions");
    ExitCode::SUCCESS
}

/// `mpm search` — look through the registry index.
fn cmd_search(registry: &str, query: Option<&str>) -> ExitCode {
    let index = match fetch_index(registry, false) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let query = query.unwrap_or("").to_ascii_lowercase();
    let mut found = 0;
    for package in &index.packages {
        if !query.is_empty() && !package.name.to_ascii_lowercase().contains(&query) {
            continue;
        }
        found += 1;
        let versions: Vec<String> = package
            .releases
            .iter()
            .map(|r| r.version.to_string())
            .collect();
        println!("{} ({})", package.name, versions.join(", "));
    }
    if found == 0 {
        println!("nothing found");
    }
    ExitCode::SUCCESS
}

fn is_valid_package_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Render a manifest table back to TOML.
///
/// The reader keeps insertion order for tables in a `BTreeMap`, so rendering
/// is deterministic but alphabetical. That is acceptable for a manifest that
/// humans edit.
fn render_manifest(table: &std::collections::BTreeMap<String, manifest::Value>) -> String {
    let mut out = String::new();

    if let Some(manifest::Value::Table(pkg)) = table.get("package") {
        out.push_str("[package]\n");
        for (key, value) in pkg {
            out.push_str(&format!("{key} = {}\n", render_value(value)));
        }
        out.push('\n');
    }

    for (name, value) in table {
        if name == "package" {
            continue;
        }
        match value {
            manifest::Value::Table(t) => {
                out.push_str(&format!("[{name}]\n"));
                for (key, v) in t {
                    out.push_str(&format!("{key} = {}\n", render_value(v)));
                }
                out.push('\n');
            }
            other => {
                out.push_str(&format!("{name} = {}\n", render_value(other)));
            }
        }
    }

    out
}

fn render_value(value: &manifest::Value) -> String {
    match value {
        manifest::Value::Str(s) => format!("\"{s}\""),
        manifest::Value::Int(i) => i.to_string(),
        manifest::Value::Float(f) => f.to_string(),
        manifest::Value::Bool(b) => b.to_string(),
        manifest::Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(render_value).collect();
            format!("[{}]", parts.join(", "))
        }
        manifest::Value::Table(t) => {
            let parts: Vec<String> = t
                .iter()
                .map(|(k, v)| format!("{k} = {}", render_value(v)))
                .collect();
            format!("{{ {} }}", parts.join(", "))
        }
    }
}

/// Parse a requirement, used by callers that already validated the text.
#[allow(dead_code)]
fn requirement(text: &str) -> Option<Requirement> {
    Requirement::parse(text).ok()
}

#[allow(dead_code)]
fn manifest_exists() -> bool {
    Path::new(&mpm::MANIFEST_FILE).exists()
}
