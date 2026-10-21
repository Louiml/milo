//! A minimal TOML reader, sufficient for `milo.toml` manifests.
//!
//! This exists so that `mpm` has **zero Cargo dependencies** and can be
//! built, audited, and vendored without a network. It supports the subset
//! of TOML that manifests actually use:
//!
//!   * comments (`#`) and blank lines
//!   * tables (`[table]`, `[table.sub]`)
//!   * key/value pairs with string, integer, float, boolean, and array values
//!   * inline tables (`{ a = "b", c = 1 }`)
//!
//! Anything outside that subset produces a clear error rather than being
//! silently mis-parsed.

use std::collections::BTreeMap;
use std::fmt;

/// A TOML value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
    Table(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Table(t) => Some(t),
            _ => None,
        }
    }
}

/// A parse failure, with the 1-based line number where it occurred.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse a TOML document into a table of tables.
pub fn parse(input: &str) -> Result<BTreeMap<String, Value>, ParseError> {
    let mut root: BTreeMap<String, Value> = BTreeMap::new();
    let mut current: Vec<String> = Vec::new();

    for (idx, raw) in input.lines().enumerate() {
        let lineno = idx + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            let name = rest.strip_suffix(']').ok_or_else(|| ParseError {
                line: lineno,
                message: "unterminated table header".to_string(),
            })?;
            current = split_key_path(name.trim(), lineno)?;
            ensure_table(&mut root, &current, lineno)?;
            continue;
        }

        let (key, value_text) = line.split_once('=').ok_or_else(|| ParseError {
            line: lineno,
            message: format!("expected `key = value`, found `{line}`"),
        })?;
        let path = split_key_path(key.trim(), lineno)?;
        let value = parse_value(value_text.trim(), lineno)?;

        let mut full = current.clone();
        full.extend(path);
        insert(&mut root, &full, value, lineno)?;
    }

    Ok(root)
}

/// Remove a trailing `#` comment, respecting quoted strings.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut quote = b'"';
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == quote {
                in_string = false;
            }
        } else if b == b'"' || b == b'\'' {
            in_string = true;
            quote = b;
        } else if b == b'#' {
            return &line[..i];
        }
        i += 1;
    }
    line
}

fn split_key_path(text: &str, line: usize) -> Result<Vec<String>, ParseError> {
    if text.is_empty() {
        return Err(ParseError {
            line,
            message: "empty key".to_string(),
        });
    }
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    for ch in text.chars() {
        match ch {
            '"' | '\'' => {
                in_quote = !in_quote;
                current.push(ch);
            }
            '.' if !in_quote => {
                parts.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    if in_quote {
        return Err(ParseError {
            line,
            message: "unterminated string in key".to_string(),
        });
    }
    parts.push(current.trim().to_string());
    if parts.iter().any(|p| p.is_empty()) {
        return Err(ParseError {
            line,
            message: "empty segment in dotted key".to_string(),
        });
    }
    Ok(parts)
}

fn ensure_table(
    root: &mut BTreeMap<String, Value>,
    path: &[String],
    line: usize,
) -> Result<(), ParseError> {
    let mut cursor = root;
    for segment in path {
        let entry = cursor
            .entry(segment.clone())
            .or_insert_with(|| Value::Table(BTreeMap::new()));
        match entry {
            Value::Table(t) => cursor = t,
            _ => {
                return Err(ParseError {
                    line,
                    message: format!("`{segment}` is not a table"),
                })
            }
        }
    }
    Ok(())
}

fn insert(
    root: &mut BTreeMap<String, Value>,
    path: &[String],
    value: Value,
    line: usize,
) -> Result<(), ParseError> {
    let (last, parents) = path.split_last().expect("path is non-empty");
    let mut cursor = root;
    for segment in parents {
        let entry = cursor
            .entry(segment.clone())
            .or_insert_with(|| Value::Table(BTreeMap::new()));
        match entry {
            Value::Table(t) => cursor = t,
            _ => {
                return Err(ParseError {
                    line,
                    message: format!("`{segment}` is not a table"),
                })
            }
        }
    }
    cursor.insert(last.clone(), value);
    Ok(())
}

fn parse_value(text: &str, line: usize) -> Result<Value, ParseError> {
    if text.is_empty() {
        return Err(ParseError {
            line,
            message: "missing value".to_string(),
        });
    }
    if let Some(rest) = text.strip_prefix('"') {
        let body = rest.strip_suffix('"').ok_or_else(|| ParseError {
            line,
            message: "unterminated string".to_string(),
        })?;
        return Ok(Value::Str(unescape(body)));
    }
    if let Some(rest) = text.strip_prefix('\'') {
        let body = rest.strip_suffix('\'').ok_or_else(|| ParseError {
            line,
            message: "unterminated literal string".to_string(),
        })?;
        return Ok(Value::Str(body.to_string()));
    }
    if text == "true" {
        return Ok(Value::Bool(true));
    }
    if text == "false" {
        return Ok(Value::Bool(false));
    }
    if let Some(inner) = text.strip_prefix('[') {
        let inner = inner.strip_suffix(']').ok_or_else(|| ParseError {
            line,
            message: "unterminated array".to_string(),
        })?;
        let mut items = Vec::new();
        for piece in split_top_level(inner) {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            items.push(parse_value(piece, line)?);
        }
        return Ok(Value::Array(items));
    }
    if let Some(inner) = text.strip_prefix('{') {
        let inner = inner.strip_suffix('}').ok_or_else(|| ParseError {
            line,
            message: "unterminated inline table".to_string(),
        })?;
        let mut table = BTreeMap::new();
        for piece in split_top_level(inner) {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            let (k, v) = piece.split_once('=').ok_or_else(|| ParseError {
                line,
                message: format!("expected `key = value` in inline table, found `{piece}`"),
            })?;
            let path = split_key_path(k.trim(), line)?;
            insert(&mut table, &path, parse_value(v.trim(), line)?, line)?;
        }
        return Ok(Value::Table(table));
    }
    if let Ok(i) = text.replace('_', "").parse::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = text.replace('_', "").parse::<f64>() {
        return Ok(Value::Float(f));
    }
    Err(ParseError {
        line,
        message: format!("cannot parse value `{text}`"),
    })
}

/// Split on commas that are not inside a string, array, or inline table.
fn split_top_level(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut quote = '"';
    let mut current = String::new();
    let mut escaped = false;

    for ch in text.chars() {
        if in_string {
            current.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                in_string = true;
                quote = ch;
                current.push(ch);
            }
            '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tables_and_scalars() {
        let doc = r#"
            # a comment
            [package]
            name = "milo-ml"
            version = "0.2.1"
            edition = 2024

            [dependencies]
            numpy = "1.26"
            "#;
        let t = parse(doc).unwrap();
        let pkg = t.get("package").unwrap().as_table().unwrap();
        assert_eq!(pkg.get("name").unwrap().as_str(), Some("milo-ml"));
        assert_eq!(pkg.get("version").unwrap().as_str(), Some("0.2.1"));
        assert_eq!(pkg.get("edition").unwrap().as_i64(), Some(2024));
        let deps = t.get("dependencies").unwrap().as_table().unwrap();
        assert_eq!(deps.get("numpy").unwrap().as_str(), Some("1.26"));
    }

    #[test]
    fn parses_arrays_and_inline_tables() {
        let doc = r#"
            [package]
            keywords = ["ai", "tensors"]
            meta = { author = "someone", stars = 12 }
            "#;
        let t = parse(doc).unwrap();
        let pkg = t.get("package").unwrap().as_table().unwrap();
        match pkg.get("keywords").unwrap() {
            Value::Array(items) => assert_eq!(items.len(), 2),
            other => panic!("expected array, got {other:?}"),
        }
        let meta = pkg.get("meta").unwrap().as_table().unwrap();
        assert_eq!(meta.get("stars").unwrap().as_i64(), Some(12));
    }

    #[test]
    fn strips_comments_inside_strings() {
        let t = parse("[package]\nname = \"a#b\"\n").unwrap();
        let pkg = t.get("package").unwrap().as_table().unwrap();
        assert_eq!(pkg.get("name").unwrap().as_str(), Some("a#b"));
    }

    #[test]
    fn reports_bad_lines() {
        let err = parse("[package\nname = \"x\"\n").unwrap_err();
        assert_eq!(err.line, 1);
    }
}
