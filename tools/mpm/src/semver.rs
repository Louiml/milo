//! Semantic versioning, and the version-requirement syntax used by
//! `milo.toml`.
//!
//! Supported requirement forms:
//!
//! ```text
//!   1.2.3      exact
//!   =1.2.3     exact
//!   ^1.2.3     compatible (>=1.2.3, <2.0.0)
//!   ~1.2.3     approximately (>=1.2.3, <1.3.0)
//!   >=1.2.3    at least
//!   >1.2.3     greater than
//!   <=1.2.3    at most
//!   *          any
//! ```

use std::cmp::Ordering;
use std::fmt;

/// A semantic version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    /// Pre-release identifiers, e.g. `["alpha", "1"]`.
    pub pre: Vec<String>,
    /// Build metadata; ignored for ordering.
    pub build: Option<String>,
}

impl Version {
    pub fn new(major: u64, minor: u64, patch: u64) -> Version {
        Version {
            major,
            minor,
            patch,
            pre: Vec::new(),
            build: None,
        }
    }

    pub fn parse(text: &str) -> Result<Version, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("empty version string".to_string());
        }

        let (rest, build) = match text.split_once('+') {
            Some((r, b)) => (r, Some(b.to_string())),
            None => (text, None),
        };
        let (core, pre_text) = match rest.split_once('-') {
            Some((c, p)) => (c, Some(p)),
            None => (rest, None),
        };

        let mut nums = core.split('.');
        let major = parse_num(nums.next(), "major")?;
        // `1` and `1.2` are accepted and mean `1.0.0` and `1.2.0`.
        let minor = match nums.next() {
            Some(part) => parse_num(Some(part), "minor")?,
            None => 0,
        };
        let patch = match nums.next() {
            Some(part) => parse_num(Some(part), "patch")?,
            None => 0,
        };
        if nums.next().is_some() {
            return Err(format!("`{text}` has too many version components"));
        }

        let pre = match pre_text {
            None => Vec::new(),
            Some(p) => p
                .split('.')
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        };

        Ok(Version {
            major,
            minor,
            patch,
            pre,
            build,
        })
    }

    fn is_prerelease(&self) -> bool {
        !self.pre.is_empty()
    }
}

fn parse_num(part: Option<&str>, what: &str) -> Result<u64, String> {
    let s = part.ok_or_else(|| format!("missing {what} version number"))?;
    s.parse::<u64>()
        .map_err(|_| format!("invalid {what} version number `{s}`"))
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() {
            write!(f, "-{}", self.pre.join("."))?;
        }
        if let Some(b) = &self.build {
            write!(f, "+{b}")?;
        }
        Ok(())
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Version) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
            .then_with(|| match (self.is_prerelease(), other.is_prerelease()) {
                // A pre-release sorts *before* its release.
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => Ordering::Equal,
                (true, true) => compare_pre(&self.pre, &other.pre),
            })
    }
}

fn compare_pre(a: &[String], b: &[String]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ord = match (x.parse::<u64>(), y.parse::<u64>()) {
            (Ok(xn), Ok(yn)) => xn.cmp(&yn),
            (Ok(_), Err(_)) => Ordering::Less,
            (Err(_), Ok(_)) => Ordering::Greater,
            (Err(_), Err(_)) => x.cmp(y),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

/// A version requirement.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    /// The comparator kind.
    pub op: Op,
    /// The bound, if the comparator has one.
    pub version: Option<Version>,
}

/// The comparator kind of a [`Requirement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Exact,
    Compatible,
    Approx,
    GreaterEq,
    Greater,
    LessEq,
    Less,
    Any,
}

impl Requirement {
    /// Parse a requirement string.
    pub fn parse(text: &str) -> Result<Requirement, String> {
        let text = text.trim();
        if text.is_empty() || text == "*" {
            return Ok(Requirement {
                op: Op::Any,
                version: None,
            });
        }
        let (op, rest) = if let Some(r) = text.strip_prefix(">=") {
            (Op::GreaterEq, r)
        } else if let Some(r) = text.strip_prefix("<=") {
            (Op::LessEq, r)
        } else if let Some(r) = text.strip_prefix('>') {
            (Op::Greater, r)
        } else if let Some(r) = text.strip_prefix('<') {
            (Op::Less, r)
        } else if let Some(r) = text.strip_prefix("==") {
            (Op::Exact, r)
        } else if let Some(r) = text.strip_prefix('=') {
            (Op::Exact, r)
        } else if let Some(r) = text.strip_prefix('^') {
            (Op::Compatible, r)
        } else if let Some(r) = text.strip_prefix('~') {
            (Op::Approx, r)
        } else {
            (Op::Exact, text)
        };

        let rest = rest.trim();
        let version = if rest.is_empty() {
            None
        } else {
            Some(Version::parse(rest)?)
        };
        if version.is_none() && op != Op::Any {
            return Err(format!("requirement `{text}` is missing a version"));
        }
        Ok(Requirement { op, version })
    }

    /// Does `version` satisfy this requirement?
    pub fn matches(&self, version: &Version) -> bool {
        let Some(bound) = &self.version else {
            return true;
        };
        match self.op {
            Op::Any => true,
            Op::Exact => version == bound,
            Op::Compatible => {
                if version < bound {
                    return false;
                }
                match bound.major {
                    0 => match bound.minor {
                        0 => version.patch == bound.patch,
                        _ => version.minor == bound.minor,
                    },
                    _ => version.major == bound.major,
                }
            }
            Op::Approx => {
                if version < bound {
                    return false;
                }
                version.major == bound.major && version.minor == bound.minor
            }
            Op::GreaterEq => version >= bound,
            Op::Greater => version > bound,
            Op::LessEq => version <= bound,
            Op::Less => version < bound,
        }
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.op, &self.version) {
            (Op::Any, _) => write!(f, "*"),
            (Op::Exact, Some(v)) => write!(f, "={v}"),
            (Op::Compatible, Some(v)) => write!(f, "^{v}"),
            (Op::Approx, Some(v)) => write!(f, "~{v}"),
            (Op::GreaterEq, Some(v)) => write!(f, ">={v}"),
            (Op::Greater, Some(v)) => write!(f, ">{v}"),
            (Op::LessEq, Some(v)) => write!(f, "<={v}"),
            (Op::Less, Some(v)) => write!(f, "<{v}"),
            (_, None) => write!(f, "*"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_versions() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.to_string(), "1.2.3");

        let p = Version::parse("1.0.0-alpha.1").unwrap();
        assert_eq!(p.pre, vec!["alpha".to_string(), "1".to_string()]);
        assert_eq!(p.to_string(), "1.0.0-alpha.1");
    }

    #[test]
    fn prereleases_sort_before_releases() {
        let a = Version::parse("1.0.0-alpha").unwrap();
        let b = Version::parse("1.0.0").unwrap();
        assert!(a < b);
    }

    #[test]
    fn caret_is_compatible() {
        let req = Requirement::parse("^1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.9.0").unwrap()));
        assert!(!req.matches(&Version::parse("2.0.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.2").unwrap()));
    }

    #[test]
    fn tilde_is_minor_locked() {
        let req = Requirement::parse("~1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn zero_major_caret_is_stricter() {
        let req = Requirement::parse("^0.2.3").unwrap();
        assert!(req.matches(&Version::parse("0.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("0.3.0").unwrap()));
    }
}
