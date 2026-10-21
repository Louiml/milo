//! The package archive: the `.mpkg` file that `mpm` downloads and
//! installs.
//!
//! The format is deliberately dull, so that it can be read and written
//! without a compression library and audited in one sitting:
//!
//! ```text
//! "MPKG1\n"                        the magic
//! <name> <version>                 the package name and version, one line each
//! <count>                          the number of files
//! for each file:
//!   <path>                         the path inside the package
//!   <size>                         the size in bytes
//!   <bytes>                        the contents
//! ```
//!
//! Every length is a decimal number on a line of its own. Paths use `/`
//! and never begin with `/` or contain `..`, so unpacking cannot write
//! outside the directory it is given.

use std::path::{Component, Path, PathBuf};

use crate::semver::Version;

/// The magic that starts every package archive.
pub const MAGIC: &str = "MPKG1";

/// The extension of a package archive.
pub const EXTENSION: &str = "mpkg";

/// One file inside a package.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// The path inside the package, using `/` as the separator.
    pub path: String,
    /// The contents.
    pub data: Vec<u8>,
}

/// A package archive.
#[derive(Debug, Clone, PartialEq)]
pub struct Package {
    pub name: String,
    pub version: Version,
    pub entries: Vec<Entry>,
}

impl Package {
    /// The files of the package, read from a directory.
    ///
    /// The manifest itself is included: a package is its own manifest.
    pub fn from_directory(name: &str, version: &Version, root: &Path) -> Result<Package, String> {
        let mut entries = Vec::new();
        collect(root, root, &mut entries)?;
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Package {
            name: name.to_string(),
            version: version.clone(),
            entries,
        })
    }

    /// The archive as bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.name.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.version.to_string().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.entries.len().to_string().as_bytes());
        out.push(b'\n');
        for entry in &self.entries {
            out.extend_from_slice(entry.path.as_bytes());
            out.push(b'\n');
            out.extend_from_slice(entry.data.len().to_string().as_bytes());
            out.push(b'\n');
            out.extend_from_slice(&entry.data);
        }
        out
    }

    /// Read an archive from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Package, String> {
        let mut cursor = Cursor { data, position: 0 };
        let magic = cursor.line()?;
        if magic != MAGIC {
            return Err(format!("this is not a package archive (it starts with `{magic}`)"));
        }
        let name = cursor.line()?;
        let version = Version::parse(&cursor.line()?)?;
        let count: usize = cursor
            .line()?
            .parse()
            .map_err(|_| "the file count is not a number".to_string())?;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let path = cursor.line()?;
            let size: usize = cursor
                .line()?
                .parse()
                .map_err(|_| "a file size is not a number".to_string())?;
            let data = cursor.take(size)?.to_vec();
            entries.push(Entry { path, data });
        }
        Ok(Package {
            name,
            version,
            entries,
        })
    }

    /// Unpack the archive into `directory`.
    pub fn unpack(&self, directory: &Path) -> Result<(), String> {
        for entry in &self.entries {
            let relative = safe_relative_path(&entry.path)?;
            let target = directory.join(&relative);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(&target, &entry.data)
                .map_err(|e| format!("{}: {e}", target.display()))?;
        }
        Ok(())
    }

    /// The manifest of the package, if it has one.
    pub fn manifest(&self) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|entry| entry.path == crate::MANIFEST_FILE)
            .map(|entry| entry.data.as_slice())
    }
}

/// Check that a path inside a package cannot escape the package.
pub fn safe_relative_path(path: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(path);
    let mut safe = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => safe.push(part),
            Component::CurDir => {}
            _ => {
                return Err(format!(
                    "`{path}` is not a safe path inside a package"
                ))
            }
        }
    }
    if safe.as_os_str().is_empty() {
        return Err("a package file has an empty path".to_string());
    }
    Ok(safe)
}

/// Collect the files of a directory, recursively.
fn collect(root: &Path, directory: &Path, entries: &mut Vec<Entry>) -> Result<(), String> {
    let listing = std::fs::read_dir(directory)
        .map_err(|e| format!("{}: {e}", directory.display()))?;
    for item in listing {
        let item = item.map_err(|e| format!("{}: {e}", directory.display()))?;
        let path = item.path();
        let name = item.file_name().to_string_lossy().into_owned();
        // Build artefacts and version control do not belong in a package.
        if name == "target" || name == ".git" || name == ".milo" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect(root, &path, entries)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| "a file is outside the package".to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            entries.push(Entry {
                path: relative,
                data,
            });
        }
    }
    Ok(())
}

struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    /// Read one line, without its line ending.
    fn line(&mut self) -> Result<String, String> {
        let start = self.position;
        while self.position < self.data.len() && self.data[self.position] != b'\n' {
            self.position += 1;
        }
        if self.position >= self.data.len() {
            return Err("the package archive ends too early".to_string());
        }
        let text = String::from_utf8_lossy(&self.data[start..self.position]).into_owned();
        self.position += 1;
        Ok(text)
    }

    /// Take `count` bytes.
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.position + count;
        if end > self.data.len() {
            return Err("the package archive ends too early".to_string());
        }
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_an_archive() {
        let package = Package {
            name: "milo-numpy".to_string(),
            version: Version::new(0, 1, 0),
            entries: vec![
                Entry {
                    path: "milo.toml".to_string(),
                    data: b"[package]\n".to_vec(),
                },
                Entry {
                    path: "src/main.milo".to_string(),
                    data: vec![0u8, 1, 2, 255],
                },
            ],
        };
        let bytes = package.to_bytes();
        let again = Package::from_bytes(&bytes).unwrap();
        assert_eq!(package, again);
    }

    #[test]
    fn rejects_a_foreign_file() {
        assert!(Package::from_bytes(b"not a package").is_err());
    }

    #[test]
    fn refuses_paths_that_escape() {
        assert!(safe_relative_path("../evil").is_err());
        assert!(safe_relative_path("/etc/passwd").is_err());
        assert!(safe_relative_path("a/../../b").is_err());
        assert_eq!(
            safe_relative_path("src/main.milo").unwrap(),
            PathBuf::from("src").join("main.milo")
        );
    }

    #[test]
    fn unpacks_into_a_directory() {
        let package = Package {
            name: "demo".to_string(),
            version: Version::new(1, 0, 0),
            entries: vec![Entry {
                path: "src/main.milo".to_string(),
                data: b"let x = 1".to_vec(),
            }],
        };
        let directory = std::env::temp_dir().join(format!(
            "mpm-unpack-{}",
            crate::sha256::digest(package.name.as_bytes()).to_hex()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        package.unpack(&directory).unwrap();
        let text = std::fs::read_to_string(directory.join("src").join("main.milo")).unwrap();
        assert_eq!(text, "let x = 1");
        let _ = std::fs::remove_dir_all(&directory);
    }
}
