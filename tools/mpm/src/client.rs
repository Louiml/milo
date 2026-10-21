//! The registry client: fetching the index and the packages it points at.
//!
//! A registry is addressed by a URL. Two forms are understood:
//!
//!   * `file:///path/to/registry` — a directory holding `index.json`.
//!     This is what the tests and the local development registry use.
//!   * `http://host/path` — a plain HTTP server. There is no TLS
//!     support, because the index is meant to be served over a local
//!     mirror or a trusted network, and a package manager that cannot
//!     verify what it fetched is not worth the convenience.
//!
//! Every download is checked against the SHA-256 in the index before it
//! is handed on.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

use crate::json;
use crate::registry::Index;

/// The name of the index file inside a local registry.
pub const INDEX_FILE: &str = "index.json";

/// A registry the client can talk to.
#[derive(Debug, Clone, PartialEq)]
pub struct Registry {
    /// The address, as the user wrote it.
    pub url: String,
}

impl Registry {
    /// A registry at an address.
    pub fn new(url: &str) -> Registry {
        Registry {
            url: url.to_string(),
        }
    }

    /// Whether the registry is a directory on this machine.
    pub fn is_local(&self) -> bool {
        !self.url.starts_with("http://") && !self.url.starts_with("https://")
    }

    /// The directory of a local registry.
    pub fn local_path(&self) -> Option<PathBuf> {
        if self.is_local() {
            Some(PathBuf::from(self.url.trim_start_matches("file://")))
        } else {
            None
        }
    }

    /// Fetch the index.
    pub fn index(&self) -> Result<Index, String> {
        let bytes = match self.local_path() {
            Some(directory) => {
                let path = directory.join(INDEX_FILE);
                std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?
            }
            None => self.get(&self.index_url())?,
        };
        let text = String::from_utf8(bytes)
            .map_err(|_| "the registry index is not valid UTF-8".to_string())?;
        let value = json::parse(&text).map_err(|e| format!("the registry index is not valid JSON: {e}"))?;
        Index::from_json(&value)
    }

    /// Fetch a package archive.
    pub fn download(&self, url: &str) -> Result<Vec<u8>, String> {
        if url.starts_with("http://") {
            self.get(url)
        } else {
            let path = PathBuf::from(url.trim_start_matches("file://"));
            std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
        }
    }

    /// The URL of the index inside this registry.
    fn index_url(&self) -> String {
        if self.url.ends_with('/') {
            format!("{}index.json", self.url)
        } else {
            format!("{}/index.json", self.url)
        }
    }

    /// A plain HTTP GET.
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| format!("`{url}` is not an HTTP address"))?;
        let (authority, path) = match rest.find('/') {
            Some(index) => (&rest[..index], &rest[index..]),
            None => (rest, "/"),
        };
        let mut stream = TcpStream::connect(authority)
            .map_err(|e| format!("{authority}: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| format!("{authority}: {e}"))?;
        let request = format!(
            "GET {path} HTTP/1.0\r\nHost: {authority}\r\nUser-Agent: mpm\r\nConnection: close\r\n\r\n"
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| format!("{authority}: {e}"))?;
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|e| format!("{authority}: {e}"))?;

        let split = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or("the server did not send a complete response")?;
        let head = String::from_utf8_lossy(&response[..split]).into_owned();
        let body = response[split + 4..].to_vec();
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("");
        if status != "200" {
            return Err(format!("{url}: the server answered {status}"));
        }
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_local_index() {
        let directory = std::env::temp_dir().join("mpm-registry-test");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let index = r#"{"protocol": 1, "packages": {}}"#;
        std::fs::write(directory.join(INDEX_FILE), index).unwrap();
        let registry = Registry::new(directory.to_str().unwrap());
        assert!(registry.is_local());
        let read = registry.index().unwrap();
        assert_eq!(read.protocol, 1);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn reports_a_missing_index() {
        let registry = Registry::new("file:///nowhere/at/all");
        assert!(registry.index().is_err());
    }
}
