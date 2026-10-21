//! SHA-256, written out here so that `mpm` keeps no dependencies.
//!
//! Every download is checked against the hash the registry index gives,
//! and the cache is keyed by it. A package manager that cannot verify
//! what it fetched is not a package manager; a hash implementation is
//! small enough to read in one sitting, so it is read in one sitting.

/// The size of a SHA-256 digest, in bytes.
pub const DIGEST_LEN: usize = 32;

/// A SHA-256 digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digest(pub [u8; DIGEST_LEN]);

impl Digest {
    /// The digest as lowercase hex.
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(DIGEST_LEN * 2);
        for byte in self.0 {
            out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
            out.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap());
        }
        out
    }

    /// Read a digest from lowercase hex.
    pub fn from_hex(text: &str) -> Result<Digest, String> {
        let text = text.trim();
        if text.len() != DIGEST_LEN * 2 {
            return Err(format!(
                "a SHA-256 digest is {} hex characters, not {}",
                DIGEST_LEN * 2,
                text.len()
            ));
        }
        let mut bytes = [0u8; DIGEST_LEN];
        for (index, byte) in bytes.iter_mut().enumerate() {
            let high = hex_digit(text.as_bytes()[index * 2])?;
            let low = hex_digit(text.as_bytes()[index * 2 + 1])?;
            *byte = (high << 4) | low;
        }
        Ok(Digest(bytes))
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

fn hex_digit(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        other => Err(format!("`{}` is not a hex digit", other as char)),
    }
}

/// The SHA-256 constants: the first thirty-two bits of the fractional
/// parts of the cube roots of the first sixty-four primes.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The SHA-256 initial state: the first thirty-two bits of the fractional
/// parts of the square roots of the first eight primes.
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// A SHA-256 hasher.
pub struct Hasher {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    /// A hasher for a new message.
    pub fn new() -> Hasher {
        Hasher {
            state: H0,
            buffer: [0u8; 64],
            buffered: 0,
            length: 0,
        }
    }

    /// Add more of the message.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let wanted = 64 - self.buffered;
            let take = wanted.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            let mut chunk = [0u8; 64];
            chunk.copy_from_slice(block);
            self.compress(&chunk);
            data = rest;
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// The digest of everything added so far.
    pub fn finish(mut self) -> Digest {
        let bits = self.length.wrapping_mul(8);
        self.update_raw(&[0x80]);
        while self.buffered != 56 {
            self.update_raw(&[0x00]);
        }
        self.update_raw(&bits.to_be_bytes());
        let mut out = [0u8; DIGEST_LEN];
        for (index, word) in self.state.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        Digest(out)
    }

    /// Add padding bytes without counting them in the length.
    fn update_raw(&mut self, data: &[u8]) {
        let saved = self.length;
        self.update(data);
        self.length = saved;
    }

    /// The compression function: one round of the message schedule and
    /// the eight working variables.
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for index in 0..16 {
            w[index] = u32::from_be_bytes([
                block[index * 4],
                block[index * 4 + 1],
                block[index * 4 + 2],
                block[index * 4 + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// The SHA-256 of a slice of bytes.
pub fn digest(data: &[u8]) -> Digest {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finish()
}

/// The SHA-256 of a file, read in chunks so that a large file does not
/// have to fit in memory.
pub fn digest_file(path: &std::path::Path) -> Result<Digest, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Hasher::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hasher.finish())
}

/// Check that `data` hashes to `expected`.
pub fn verify(data: &[u8], expected: &str) -> Result<(), String> {
    let actual = digest(data);
    if actual.to_hex() == expected.trim().to_ascii_lowercase() {
        Ok(())
    } else {
        Err(format!(
            "the file does not match its hash: expected {expected}, got {actual}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_empty_string() {
        let d = digest(b"");
        assert_eq!(
            d.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn matches_the_standard_vectors() {
        assert_eq!(
            digest(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").to_hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn handles_long_messages() {
        let data = vec![b'a'; 1_000_000];
        assert_eq!(
            digest(&data).to_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn round_trips_hex() {
        let d = digest(b"milo");
        assert_eq!(Digest::from_hex(&d.to_hex()).unwrap(), d);
        assert!(Digest::from_hex("not hex").is_err());
        assert!(Digest::from_hex("ab").is_err());
    }

    #[test]
    fn verifies_what_it_should() {
        let d = digest(b"milo");
        assert!(verify(b"milo", &d.to_hex()).is_ok());
        assert!(verify(b"not milo", &d.to_hex()).is_err());
    }
}
