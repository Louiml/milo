//! The lockfile: the exact versions that were installed.
//!
//! `mpm install` writes `milo.lock` next to the manifest. It records
//! what was resolved, not what was asked for, so that a later install
//! fetches exactly the same bytes — the ones whose hash is in the file.

use std::path::Path;

use crate::json::{self, Json};
use crate::semver::Version;

/// The name of the lockfile.
pub const FILE: &str = "milo.lock";

/// One package in the lockfile.
#[derive(Debug, Clone, PartialEq)]
pub struct LockedPackage {
    pub name: String,
    pub version: Version,
    /// The SHA-256 of the archive that was installed.
    pub sha256: String,
    /// Where the archive came from.
    pub source: String,
}

/// The contents of a lockfile.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lockfile {
    /// The protocol version of the lockfile format.
    pub version: u32,
    pub packages: Vec<LockedPackage>,
}

impl Lockfile {
    /// The lockfile as JSON.
    pub fn to_json(&self) -> Json {
        let mut packages = Vec::new();
        for package in &self.packages {
            let mut entry = std::collections::BTreeMap::new();
            entry.insert("name".to_string(), Json::String(package.name.clone()));
            entry.insert(
                "version".to_string(),
                Json::String(package.version.to_string()),
            );
            entry.insert("sha256".to_string(), Json::String(package.sha256.clone()));
            entry.insert("source".to_string(), Json::String(package.source.clone()));
            packages.push(Json::Object(entry));
        }
        let mut root = std::collections::BTreeMap::new();
        root.insert("lockfile".to_string(), Json::Number(self.version as f64));
        root.insert("packages".to_string(), Json::Array(packages));
        Json::Object(root)
    }

    /// Read a lockfile from JSON.
    pub fn from_json(value: &Json) -> Result<Lockfile, String> {
        let version = value
            .get("lockfile")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let mut packages = Vec::new();
        if let Some(items) = value.get("packages").and_then(|v| v.as_array()) {
            for item in items {
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("a locked package has no name")?
                    .to_string();
                let version = Version::parse(
                    item.get("version")
                        .and_then(|v| v.as_str())
                        .ok_or("a locked package has no version")?,
                )?;
                let sha256 = item
                    .get("sha256")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let source = item
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                packages.push(LockedPackage {
                    name,
                    version,
                    sha256,
                    source,
                });
            }
        }
        Ok(Lockfile { version, packages })
    }

    /// Read a lockfile from a file.
    pub fn read(path: &Path) -> Result<Lockfile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value = json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Lockfile::from_json(&value).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Write the lockfile to a file.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let text = json::to_string_pretty(&self.to_json());
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The locked version of a package, if it is locked.
    pub fn find(&self, name: &str) -> Option<&LockedPackage> {
        self.packages.iter().find(|p| p.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let lockfile = Lockfile {
            version: 1,
            packages: vec![LockedPackage {
                name: "milo-numpy".to_string(),
                version: Version::new(0, 1, 0),
                sha256: "abc".to_string(),
                source: "https://example.invalid/index.json".to_string(),
            }],
        };
        let text = json::to_string_pretty(&lockfile.to_json());
        let value = json::parse(&text).unwrap();
        let again = Lockfile::from_json(&value).unwrap();
        assert_eq!(lockfile, again);
    }

    #[test]
    fn finds_a_package() {
        let lockfile = Lockfile {
            version: 1,
            packages: vec![LockedPackage {
                name: "milo-numpy".to_string(),
                version: Version::new(0, 1, 0),
                sha256: "abc".to_string(),
                source: String::new(),
            }],
        };
        assert!(lockfile.find("milo-numpy").is_some());
        assert!(lockfile.find("other").is_none());
    }
}
