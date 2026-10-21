//! The package registry: addressing, the index format, and the protocol
//! version.
//!
//! ## Protocol v1
//!
//! A registry is a static index served over HTTP or checked out from Git.
//! There is no mutation API in v1: publishing pushes a new commit (Git) or
//! uploads a new immutable index file (HTTP). This keeps the reference
//! implementation trivial to audit and impossible to attack with
//! confused-deputy writes.
//!
//! ```text
//! GET /index.json
//! {
//!   "protocol": 1,
//!   "packages": {
//!     "milo-numpy": {
//!       "versions": {
//!         "0.1.0": {
//!           "sha256": "…",
//!           "size": 12345,
//!           "url": "https://…/milo-numpy-0.1.0.mpkg",
//!           "signature": "…"
//!         }
//!       }
//!     }
//!   }
//! }
//! ```
//!
//! Every download is verified against `sha256` before it is unpacked, and
//! against `signature` when the index carries one.

use crate::semver::{Requirement, Version};

/// The registry protocol version this client understands.
pub const PROTOCOL_VERSION: u32 = 1;

/// A package reference: a name and, optionally, a version requirement.
#[derive(Debug, Clone, PartialEq)]
pub struct PackageSpec {
    pub name: String,
    pub requirement: Requirement,
}

impl PackageSpec {
    /// Parse a spec of the form `name`, `name@1.2.3`, or `name@^1.2`.
    pub fn parse(text: &str) -> Result<PackageSpec, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("empty package name".to_string());
        }
        let (name, requirement) = match text.split_once('@') {
            Some((n, r)) => (n.trim(), Requirement::parse(r)?),
            None => (text, Requirement::parse("*")?),
        };
        validate_name(name)?;
        Ok(PackageSpec {
            name: name.to_string(),
            requirement,
        })
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("package name is empty".to_string());
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() {
        return Err(format!("package name `{name}` must start with a letter"));
    }
    for c in name.chars() {
        if !(c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(format!(
                "package name `{name}` may only contain letters, digits, `-` and `_`"
            ));
        }
    }
    Ok(())
}

/// One published version of a package.
#[derive(Debug, Clone, PartialEq)]
pub struct PackageRelease {
    pub version: Version,
    /// Lowercase hex SHA-256 of the package archive.
    pub sha256: String,
    /// Archive size in bytes.
    pub size: u64,
    /// Where to download it from.
    pub url: String,
    /// Optional detached signature.
    pub signature: Option<String>,
    /// The packages this one needs, and the versions it allows.
    pub dependencies: Vec<(String, Requirement)>,
}

/// A named package with all of its published versions.
#[derive(Debug, Clone, PartialEq)]
pub struct PackageEntry {
    pub name: String,
    pub releases: Vec<PackageRelease>,
}

impl PackageEntry {
    /// The newest release satisfying `requirement`.
    pub fn best_match(&self, requirement: &Requirement) -> Option<&PackageRelease> {
        self.releases
            .iter()
            .filter(|r| requirement.matches(&r.version))
            .max_by(|a, b| a.version.cmp(&b.version))
    }
}

/// The registry index: protocol version plus the package table.
#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    pub protocol: u32,
    pub packages: Vec<PackageEntry>,
}

impl Index {
    pub fn find(&self, name: &str) -> Option<&PackageEntry> {
        self.packages.iter().find(|p| p.name == name)
    }

    /// Read an index from JSON.
    pub fn from_json(value: &crate::json::Json) -> Result<Index, String> {
        let protocol = value
            .get("protocol")
            .and_then(|v| v.as_u64())
            .unwrap_or(PROTOCOL_VERSION as u64) as u32;
        if protocol != PROTOCOL_VERSION {
            return Err(format!(
                "this registry speaks protocol {protocol}, but this client speaks {PROTOCOL_VERSION}"
            ));
        }
        let mut packages = Vec::new();
        let table = value
            .get("packages")
            .and_then(|v| v.as_object())
            .ok_or("the index has no `packages` table")?;
        for (name, entry) in table {
            let mut releases = Vec::new();
            let versions = entry
                .get("versions")
                .and_then(|v| v.as_object())
                .ok_or_else(|| format!("`{name}` has no `versions` table"))?;
            for (version_text, release) in versions {
                let version = Version::parse(version_text)?;
                let sha256 = release
                    .get("sha256")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("`{name} {version_text}` has no `sha256`"))?
                    .to_string();
                let size = release.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                let url = release
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("`{name} {version_text}` has no `url`"))?
                    .to_string();
                let signature = release
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let mut dependencies = Vec::new();
                if let Some(deps) = release.get("dependencies").and_then(|v| v.as_object()) {
                    for (dep_name, requirement) in deps {
                        let text = requirement.as_str().unwrap_or("*");
                        dependencies.push((dep_name.clone(), Requirement::parse(text)?));
                    }
                }
                releases.push(PackageRelease {
                    version,
                    sha256,
                    size,
                    url,
                    signature,
                    dependencies,
                });
            }
            releases.sort_by(|a, b| a.version.cmp(&b.version));
            packages.push(PackageEntry {
                name: name.clone(),
                releases,
            });
        }
        packages.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Index { protocol, packages })
    }

    /// The index as JSON.
    pub fn to_json(&self) -> crate::json::Json {
        use crate::json::Json;
        let mut table = std::collections::BTreeMap::new();
        for package in &self.packages {
            let mut versions = std::collections::BTreeMap::new();
            for release in &package.releases {
                let mut entry = std::collections::BTreeMap::new();
                entry.insert("sha256".to_string(), Json::String(release.sha256.clone()));
                entry.insert("size".to_string(), Json::Number(release.size as f64));
                entry.insert("url".to_string(), Json::String(release.url.clone()));
                if let Some(signature) = &release.signature {
                    entry.insert("signature".to_string(), Json::String(signature.clone()));
                }
                if !release.dependencies.is_empty() {
                    let mut deps = std::collections::BTreeMap::new();
                    for (name, requirement) in &release.dependencies {
                        deps.insert(name.clone(), Json::String(requirement.to_string()));
                    }
                    entry.insert("dependencies".to_string(), Json::Object(deps));
                }
                versions.insert(release.version.to_string(), Json::Object(entry));
            }
            let mut package_entry = std::collections::BTreeMap::new();
            package_entry.insert("versions".to_string(), Json::Object(versions));
            table.insert(package.name.clone(), Json::Object(package_entry));
        }
        let mut root = std::collections::BTreeMap::new();
        root.insert("protocol".to_string(), Json::Number(self.protocol as f64));
        root.insert("packages".to_string(), Json::Object(table));
        Json::Object(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_specs() {
        let s = PackageSpec::parse("milo-numpy").unwrap();
        assert_eq!(s.name, "milo-numpy");
        assert_eq!(s.requirement, Requirement::parse("*").unwrap());

        let s = PackageSpec::parse("milo-numpy@^0.2").unwrap();
        assert_eq!(s.requirement.to_string(), "^0.2.0");
    }

    #[test]
    fn rejects_bad_names() {
        assert!(PackageSpec::parse("1bad").is_err());
        assert!(PackageSpec::parse("has space").is_err());
    }
}
