//! A minimal JSON reader and writer.
//!
//! The registry index and the lockfile are JSON. `mpm` has no Cargo
//! dependencies, so the reader is written out here. It understands the
//! whole of JSON except numbers with exponents that do not fit a `f64`,
//! which it reports rather than guessing at.

use std::collections::BTreeMap;
use std::fmt;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().and_then(|n| if n >= 0.0 { Some(n as u64) } else { None })
    }

    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object().and_then(|map| map.get(key))
    }

    /// A short name for the kind of value, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Number(_) => "a number",
            Json::String(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

/// A parse failure.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse a whole JSON document.
pub fn parse(text: &str) -> Result<Json, ParseError> {
    let mut reader = Reader {
        bytes: text.as_bytes(),
        position: 0,
    };
    reader.skip_whitespace();
    let value = reader.value()?;
    reader.skip_whitespace();
    if reader.position != reader.bytes.len() {
        return Err(ParseError {
            message: format!("trailing text at byte {}", reader.position),
        });
    }
    Ok(value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek() == Some(byte) {
            self.position += 1;
            Ok(())
        } else {
            Err(ParseError {
                message: format!(
                    "expected `{}` at byte {}",
                    byte as char, self.position
                ),
            })
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.peek() {
            None => Err(ParseError {
                message: "the document ends too early".to_string(),
            }),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => {
                self.keyword("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.keyword("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.keyword("null")?;
                Ok(Json::Null)
            }
            Some(_) => self.number(),
        }
    }

    fn keyword(&mut self, word: &str) -> Result<(), ParseError> {
        if self.bytes[self.position..].starts_with(word.as_bytes()) {
            self.position += word.len();
            Ok(())
        } else {
            Err(ParseError {
                message: format!("expected `{word}` at byte {}", self.position),
            })
        }
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.position += 1;
                }
                Some(b'}') => {
                    self.position += 1;
                    break;
                }
                _ => {
                    return Err(ParseError {
                        message: format!("expected `,` or `}}` at byte {}", self.position),
                    })
                }
            }
        }
        Ok(Json::Object(map))
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.position += 1;
                }
                Some(b']') => {
                    self.position += 1;
                    break;
                }
                _ => {
                    return Err(ParseError {
                        message: format!("expected `,` or `]` at byte {}", self.position),
                    })
                }
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self.peek().ok_or_else(|| ParseError {
                message: "the string ends too early".to_string(),
            })?;
            self.position += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = self.peek().ok_or_else(|| ParseError {
                        message: "the string ends too early".to_string(),
                    })?;
                    self.position += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.hex4()?;
                            // A surrogate pair, if there is one.
                            if (0xd800..0xdc00).contains(&code) {
                                if self.peek() == Some(b'\\') {
                                    self.position += 1;
                                    self.expect(b'u')?;
                                    let low = self.hex4()?;
                                    let combined =
                                        0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                                    out.push(
                                        char::from_u32(combined).ok_or_else(|| ParseError {
                                            message: "a bad surrogate pair".to_string(),
                                        })?,
                                    );
                                } else {
                                    out.push('\u{fffd}');
                                }
                            } else {
                                out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                            }
                        }
                        other => {
                            return Err(ParseError {
                                message: format!("`\\{}` is not an escape", other as char),
                            })
                        }
                    }
                }
                _ => {
                    // Copy the UTF-8 sequence through unchanged.
                    let start = self.position - 1;
                    let length = utf8_length(byte);
                    let end = (start + length).min(self.bytes.len());
                    self.position = end;
                    match std::str::from_utf8(&self.bytes[start..end]) {
                        Ok(text) => out.push_str(text),
                        Err(_) => {
                            return Err(ParseError {
                                message: format!("bad UTF-8 at byte {start}"),
                            })
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        if self.position + 4 > self.bytes.len() {
            return Err(ParseError {
                message: "a `\\u` escape needs four hex digits".to_string(),
            });
        }
        let text = std::str::from_utf8(&self.bytes[self.position..self.position + 4])
            .map_err(|_| ParseError {
                message: "a `\\u` escape needs four hex digits".to_string(),
            })?;
        let code = u32::from_str_radix(text, 16).map_err(|_| ParseError {
            message: format!("`{text}` is not a hex escape"),
        })?;
        self.position += 4;
        Ok(code)
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        while let Some(byte) = self.peek() {
            if byte.is_ascii_digit() || byte == b'.' || byte == b'e' || byte == b'E' {
                self.position += 1;
            } else if (byte == b'+' || byte == b'-')
                && matches!(self.bytes[self.position - 1], b'e' | b'E')
            {
                self.position += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position]).map_err(|_| ParseError {
            message: format!("bad number at byte {start}"),
        })?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| ParseError {
                message: format!("`{text}` is not a number"),
            })
    }
}

fn utf8_length(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Render a value as JSON, with two spaces of indentation.
pub fn to_string_pretty(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out.push('\n');
    out
}

fn write_value(out: &mut String, value: &Json, depth: usize) {
    let pad = "  ".repeat(depth);
    let inner_pad = "  ".repeat(depth + 1);
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 9e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{n}"));
            }
        }
        Json::String(s) => write_string(out, s),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                out.push_str(&inner_pad);
                write_value(out, item, depth + 1);
                if index + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad);
            out.push(']');
        }
        Json::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            let count = map.len();
            for (index, (key, item)) in map.iter().enumerate() {
                out.push_str(&inner_pad);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1);
                if index + 1 < count {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad);
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_the_index_uses() {
        let text = r#"{
            "protocol": 1,
            "packages": {
                "milo-numpy": {
                    "versions": {
                        "0.1.0": {
                            "sha256": "abcd",
                            "size": 1234,
                            "url": "https://example.invalid/milo-numpy-0.1.0.mpkg"
                        }
                    }
                }
            }
        }"#;
        let value = parse(text).unwrap();
        assert_eq!(value.get("protocol").unwrap().as_f64(), Some(1.0));
        let package = value.get("packages").unwrap().get("milo-numpy").unwrap();
        let release = package.get("versions").unwrap().get("0.1.0").unwrap();
        assert_eq!(release.get("sha256").unwrap().as_str(), Some("abcd"));
        assert_eq!(release.get("size").unwrap().as_u64(), Some(1234));
    }

    #[test]
    fn parses_escapes() {
        let value = parse(r#""a\nbAé""#).unwrap();
        assert_eq!(value.as_str(), Some("a\nbA\u{e9}"));
    }

    #[test]
    fn rejects_bad_documents() {
        assert!(parse("{").is_err());
        assert!(parse("{\"a\": 1,}").is_err());
        assert!(parse("[1] extra").is_err());
        assert!(parse("nul").is_err());
    }

    #[test]
    fn round_trips() {
        let value = parse(r#"{"a": [1, 2, 3], "b": {"c": true}}"#).unwrap();
        let text = to_string_pretty(&value);
        let again = parse(&text).unwrap();
        assert_eq!(value, again);
    }
}
