//! The resolver: turning the requirements in a manifest into the exact
//! set of packages to install.
//!
//! The resolver is deliberately simple and strict:
//!
//!   * every requirement must be satisfied by exactly one version;
//!   * a package that is needed twice must be needed at the same
//!     version both times — otherwise the answer would depend on the
//!     order the packages were visited in, and a lockfile that changes
//!     when the manifest is reordered is not a lockfile;
//!   * a cycle is reported, not followed.

use crate::registry::Index;
use crate::semver::{Requirement, Version};

/// One resolved package.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub name: String,
    pub version: Version,
    /// The SHA-256 of the archive, as the index gives it.
    pub sha256: String,
    /// Where to download it from.
    pub url: String,
}

/// A resolution failure, with a message that says what to do.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveError {
    pub message: String,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ResolveError {}

/// Resolve `requirements` against `index`.
///
/// The result is sorted by name, so the same inputs always give the
/// same answer.
pub fn resolve(
    requirements: &[(String, Requirement)],
    index: &Index,
) -> Result<Vec<Resolved>, ResolveError> {
    let mut chosen: Vec<Resolved> = Vec::new();
    let mut pending: Vec<(String, Requirement)> = requirements.to_vec();
    let mut seen: Vec<String> = Vec::new();

    while let Some((name, requirement)) = pending.pop() {
        if let Some(existing) = chosen.iter().find(|p| p.name == name) {
            // Already resolved: the new requirement has to agree with the
            // version that was chosen.
            if !requirement.matches(&existing.version) {
                return Err(ResolveError {
                    message: format!(
                        "`{name}` is needed as {} and as {}, and no version of it satisfies both",
                        existing.version,
                        requirement
                    ),
                });
            }
            continue;
        }

        let entry = index.find(&name).ok_or_else(|| ResolveError {
            message: format!("the registry does not have a package called `{name}`"),
        })?;
        let release = entry.best_match(&requirement).ok_or_else(|| ResolveError {
            message: format!(
                "the registry has no version of `{name}` that matches {requirement} (it has {})",
                describe_versions(entry)
            ),
        })?;

        if seen.contains(&name) {
            return Err(ResolveError {
                message: format!("the dependencies of `{name}` form a cycle"),
            });
        }
        seen.push(name.clone());

        for (dependency, requirement) in &release.dependencies {
            pending.push((dependency.clone(), requirement.clone()));
        }

        chosen.push(Resolved {
            name: name.clone(),
            version: release.version.clone(),
            sha256: release.sha256.clone(),
            url: release.url.clone(),
        });
    }

    chosen.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(chosen)
}

/// The versions a package has, for an error message.
fn describe_versions(entry: &crate::registry::PackageEntry) -> String {
    if entry.releases.is_empty() {
        return "none".to_string();
    }
    entry
        .releases
        .iter()
        .map(|r| r.version.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{PackageEntry, PackageRelease};

    fn index() -> Index {
        Index {
            protocol: 1,
            packages: vec![
                PackageEntry {
                    name: "milo-core".to_string(),
                    releases: vec![PackageRelease {
                        version: Version::new(0, 1, 0),
                        sha256: "a".to_string(),
                        size: 1,
                        url: "core".to_string(),
                        signature: None,
                        dependencies: Vec::new(),
                    }],
                },
                PackageEntry {
                    name: "milo-numpy".to_string(),
                    releases: vec![
                        PackageRelease {
                            version: Version::new(0, 1, 0),
                            sha256: "b".to_string(),
                            size: 1,
                            url: "numpy-0.1.0".to_string(),
                            signature: None,
                            dependencies: vec![(
                                "milo-core".to_string(),
                                Requirement::parse("^0.1").unwrap(),
                            )],
                        },
                        PackageRelease {
                            version: Version::new(0, 2, 0),
                            sha256: "c".to_string(),
                            size: 1,
                            url: "numpy-0.2.0".to_string(),
                            signature: None,
                            dependencies: Vec::new(),
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn resolves_the_whole_tree() {
        let requirements = vec![(
            "milo-numpy".to_string(),
            Requirement::parse("^0.1").unwrap(),
        )];
        let resolved = resolve(&requirements, &index()).unwrap();
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "milo-core");
        assert_eq!(resolved[1].name, "milo-numpy");
        // `^0.1` allows 0.1.x only: a zero major version is strict about
        // the minor version.
        assert_eq!(resolved[1].version.to_string(), "0.1.0");
    }

    #[test]
    fn reports_a_missing_package() {
        let requirements = vec![("nope".to_string(), Requirement::parse("*").unwrap())];
        let error = resolve(&requirements, &index()).unwrap_err();
        assert!(error.message.contains("nope"));
    }

    #[test]
    fn reports_a_version_that_does_not_match() {
        let requirements = vec![(
            "milo-numpy".to_string(),
            Requirement::parse("^2.0").unwrap(),
        )];
        let error = resolve(&requirements, &index()).unwrap_err();
        assert!(error.message.contains("no version"));
    }
}
