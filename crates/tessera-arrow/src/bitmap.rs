//! Validity / boolean bitmaps with LSB bit numbering (Arrow spec).
use crate::buffer::{Buffer, MutableBuffer};
use crate::error::{ArrowError, Result};

#[inline]
fn get_bit(bytes: &[u8], i: usize) -> bool {
    bytes[i >> 3] & (1 << (i & 7)) != 0
}

/// An immutable bitmap: a byte [`Buffer`] plus a bit offset and bit length.
#[derive(Clone, Debug)]
pub struct Bitmap {
    bytes: Buffer<u8>,
    offset: usize,
    len: usize,
    unset: usize,
}

impl Bitmap {
    /// Build from raw bytes; `len` bits starting at bit 0.
    pub fn try_new(bytes: Buffer<u8>, len: usize) -> Result<Self> {
        if len.div_ceil(8) > bytes.len() {
            return Err(ArrowError::LengthMismatch(format!(
                "bitmap of {len} bits needs {} bytes, got {}",
                len.div_ceil(8),
                bytes.len()
            )));
        }
        let mut b = Bitmap {
            bytes,
            offset: 0,
            len,
            unset: 0,
        };
        b.unset = len - b.count_set();
        Ok(b)
    }

    /// Bitmap from an iterator of bools.
    pub fn from_bools<I: IntoIterator<Item = bool>>(it: I) -> Self {
        let mut m = MutableBitmap::new();
        for b in it {
            m.push(b);
        }
        m.into_bitmap()
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether there are no bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Number of zero bits (nulls when used as validity).
    pub fn unset_bits(&self) -> usize {
        self.unset
    }

    /// Bit `i`. Panics (in debug and release) if out of bounds, like slice indexing.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        get_bit(&self.bytes, self.offset + i)
    }

    /// The 64 bits starting at logical bit `start` (bits past `len` are zero).
    #[inline]
    pub fn word(&self, start: usize) -> u64 {
        let n = (self.len - start.min(self.len)).min(64);
        if n == 0 {
            return 0;
        }
        let bit = self.offset + start;
        let byte = bit >> 3;
        let shift = bit & 7;
        let mut buf = [0u8; 16];
        let avail = (self.bytes.len() - byte).min(9);
        buf[..avail].copy_from_slice(&self.bytes[byte..byte + avail]);
        let lo = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let mut w = lo >> shift;
        if shift > 0 {
            w |= (buf[8] as u64) << (64 - shift);
        }
        if n < 64 {
            w &= (1u64 << n) - 1;
        }
        w
    }

    /// Iterator over 64-bit words covering the bitmap.
    pub fn words(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len.div_ceil(64)).map(move |i| self.word(i * 64))
    }

    /// Count of set bits.
    pub fn count_set(&self) -> usize {
        self.words().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate bits.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }

    /// Zero-copy slice.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        if offset.checked_add(len).is_none_or(|e| e > self.len) {
            return Err(ArrowError::OutOfBounds(format!(
                "bitmap slice {offset}+{len} of {}",
                self.len
            )));
        }
        let mut b = Bitmap {
            bytes: self.bytes.clone(),
            offset: self.offset + offset,
            len,
            unset: 0,
        };
        b.unset = len - b.count_set();
        Ok(b)
    }

    /// Build from 64-bit words (LSB first).
    pub fn from_words(words: &[u64], len: usize) -> Self {
        let mut m = MutableBuffer::<u8>::with_capacity(words.len() * 8);
        for w in words {
            m.extend_from_slice(&w.to_le_bytes());
        }
        m.truncate(len.div_ceil(8));
        // Mask stray bits past len in the last byte to keep the bitmap canonical.
        if len % 8 != 0 {
            if let Some(last) = m.as_mut_slice().last_mut() {
                *last &= (1u8 << (len % 8)) - 1;
            }
        }
        let bytes = m.into_buffer();
        let mut b = Bitmap {
            bytes,
            offset: 0,
            len,
            unset: 0,
        };
        b.unset = len - b.count_set();
        b
    }

    fn binary(&self, other: &Bitmap, f: impl Fn(u64, u64) -> u64) -> Result<Self> {
        if self.len != other.len {
            return Err(ArrowError::LengthMismatch(format!(
                "{} vs {}",
                self.len, other.len
            )));
        }
        let words: Vec<u64> = self
            .words()
            .zip(other.words())
            .map(|(a, b)| f(a, b))
            .collect();
        Ok(Self::from_words(&words, self.len))
    }

    /// Bitwise AND.
    pub fn and(&self, other: &Bitmap) -> Result<Self> {
        self.binary(other, |a, b| a & b)
    }
    /// Bitwise OR.
    pub fn or(&self, other: &Bitmap) -> Result<Self> {
        self.binary(other, |a, b| a | b)
    }
    /// Bitwise NOT.
    pub fn not(&self) -> Self {
        let words: Vec<u64> = self.words().map(|w| !w).collect();
        Self::from_words(&words, self.len)
    }
    /// Underlying byte buffer and bit offset.
    pub fn as_raw(&self) -> (&Buffer<u8>, usize) {
        (&self.bytes, self.offset)
    }
}

impl PartialEq for Bitmap {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.words().eq(other.words())
    }
}

/// A growable bitmap.
#[derive(Default)]
pub struct MutableBitmap {
    bytes: MutableBuffer<u8>,
    len: usize,
}

impl MutableBitmap {
    /// Empty bitmap.
    pub fn new() -> Self {
        Self::default()
    }
    /// With capacity for `bits` bits.
    pub fn with_capacity(bits: usize) -> Self {
        MutableBitmap {
            bytes: MutableBuffer::with_capacity(bits.div_ceil(8)),
            len: 0,
        }
    }
    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append a bit.
    #[inline]
    pub fn push(&mut self, v: bool) {
        if self.len % 8 == 0 {
            self.bytes.push(0);
        }
        if v {
            let last = self.bytes.len() - 1;
            self.bytes.as_mut_slice()[last] |= 1 << (self.len % 8);
        }
        self.len += 1;
    }
    /// Append `n` copies of `v`.
    pub fn extend_constant(&mut self, n: usize, v: bool) {
        for _ in 0..n {
            self.push(v);
        }
    }
    /// Freeze.
    pub fn into_bitmap(self) -> Bitmap {
        let len = self.len;
        let bytes = self.bytes.into_buffer();
        let mut b = Bitmap {
            bytes,
            offset: 0,
            len,
            unset: 0,
        };
        b.unset = len - b.count_set();
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_count() {
        let bits: Vec<bool> = (0..200).map(|i| i % 3 == 0).collect();
        let b = Bitmap::from_bools(bits.iter().copied());
        assert_eq!(b.len(), 200);
        assert_eq!(b.iter().collect::<Vec<_>>(), bits);
        assert_eq!(b.count_set(), bits.iter().filter(|x| **x).count());
        assert_eq!(b.unset_bits(), 200 - b.count_set());
    }

    #[test]
    fn unaligned_slice_words() {
        let bits: Vec<bool> = (0..300).map(|i| (i * 7) % 5 < 2).collect();
        let b = Bitmap::from_bools(bits.iter().copied());
        for off in [0, 1, 3, 7, 8, 63, 64, 65, 130] {
            let len = 300 - off - 5;
            let s = b.slice(off, len).unwrap();
            assert_eq!(s.iter().collect::<Vec<_>>(), &bits[off..off + len]);
            let from_words: Vec<bool> = (0..len)
                .map(|i| s.word(i / 64 * 64) >> (i % 64) & 1 == 1)
                .collect();
            assert_eq!(from_words, &bits[off..off + len]);
            assert_eq!(
                s.unset_bits(),
                bits[off..off + len].iter().filter(|x| !**x).count()
            );
        }
        assert!(b.slice(290, 20).is_err());
    }

    #[test]
    fn logical_ops() {
        let a = Bitmap::from_bools((0..70).map(|i| i % 2 == 0));
        let b = Bitmap::from_bools((0..70).map(|i| i % 3 == 0));
        let and = a.and(&b).unwrap();
        let or = a.or(&b).unwrap();
        let not = a.not();
        for i in 0..70 {
            assert_eq!(and.get(i), i % 6 == 0);
            assert_eq!(or.get(i), i % 2 == 0 || i % 3 == 0);
            assert_eq!(not.get(i), i % 2 == 1);
        }
        assert_eq!(not.len(), 70);
        assert_eq!(not.count_set(), 35);
        assert!(a.and(&Bitmap::from_bools([true])).is_err());
    }
}
