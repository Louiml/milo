//! The package cache: the place where downloaded archives are kept.
//!
//! The cache is keyed by the hash of the archive, so a package that was
//! verified once is never downloaded or unpacked twice. It is also what
//! `--offline` reads from: with no network, everything that is needed
//! has to be in the cache already.

use std::path::{Path, PathBuf};

use crate::package::Package;
use crate::sha256;

/// The name of the cache directory, under the user's home directory.
pub const CACHE_DIRECTORY: &str = ".milo";

/// The cache of downloaded packages.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// The cache in the user's home directory.
    pub fn default_location() -> Result<Cache, String> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or("there is no home directory, so there is no cache")?;
        let root = PathBuf::from(home).join(CACHE_DIRECTORY).join("cache");
        Ok(Cache { root })
    }

    /// A cache in a given directory.
    pub fn at(root: PathBuf) -> Cache {
        Cache { root }
    }

    /// The directory the cache keeps its archives in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the archive with a given hash is kept.
    pub fn path_for_hash(&self, hash: &str) -> PathBuf {
        self.root.join(format!("{hash}.mpkg"))
    }

    /// Whether the cache holds an archive with this hash.
    pub fn contains(&self, hash: &str) -> bool {
        self.path_for_hash(hash).is_file()
    }

    /// Put a package into the cache and return its hash.
    pub fn store(&self, package: &Package) -> Result<String, String> {
        let bytes = package.to_bytes();
        let hash = sha256::digest(&bytes).to_hex();
        std::fs::create_dir_all(&self.root)
            .map_err(|e| format!("{}: {e}", self.root.display()))?;
        let path = self.path_for_hash(&hash);
        if !path.is_file() {
            std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(hash)
    }

    /// Read a package out of the cache.
    pub fn load(&self, hash: &str) -> Result<Package, String> {
        let path = self.path_for_hash(hash);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let actual = sha256::digest(&bytes).to_hex();
        if actual != hash {
            return Err(format!(
                "the cached package {} does not match its hash (expected {hash}, got {actual})",
                path.display()
            ));
        }
        Package::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semver::Version;

    fn sample() -> Package {
        Package {
            name: "demo".to_string(),
            version: Version::new(1, 0, 0),
            entries: vec![crate::package::Entry {
                path: "milo.toml".to_string(),
                data: b"[package]\n".to_vec(),
            }],
        }
    }

    #[test]
    fn stores_and_loads() {
        let directory = std::env::temp_dir().join("mpm-cache-test");
        let _ = std::fs::remove_dir_all(&directory);
        let cache = Cache::at(directory.clone());
        let hash = cache.store(&sample()).unwrap();
        assert!(cache.contains(&hash));
        let loaded = cache.load(&hash).unwrap();
        assert_eq!(loaded.name, "demo");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn refuses_a_package_that_does_not_match() {
        let directory = std::env::temp_dir().join("mpm-cache-test-bad");
        let _ = std::fs::remove_dir_all(&directory);
        let cache = Cache::at(directory.clone());
        let hash = cache.store(&sample()).unwrap();
        std::fs::write(cache.path_for_hash(&hash), b"not a package").unwrap();
        assert!(cache.load(&hash).is_err());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
