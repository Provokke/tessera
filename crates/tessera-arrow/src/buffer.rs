//! 64-byte aligned, reference-counted, zero-copy-sliceable buffers.
//!
//! [`MutableBuffer`] owns a growable allocation that is always aligned to
//! [`ALIGNMENT`] bytes and padded to a multiple of [`ALIGNMENT`]. Freezing it
//! into a [`Buffer`] moves the allocation into an `Arc` without copying;
//! [`Buffer::slice`] then shares that allocation.
use std::alloc::{self, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::error::{ArrowError, Result};

/// Alignment (and padding granularity) of every allocation, per the Arrow spec
/// recommendation for SIMD-friendly buffers.
pub const ALIGNMENT: usize = 64;

mod sealed {
    pub trait Sealed {}
}

/// Fixed-width plain-old-data element types that may live in a [`Buffer`].
///
/// # Safety
/// Implementors must be `Copy`, have no padding, no invalid bit patterns
/// (every bit pattern, including all zeros, is a valid value) and an
/// alignment that divides [`ALIGNMENT`].
pub unsafe trait NativeType:
    sealed::Sealed + Copy + Send + Sync + Default + PartialEq + fmt::Debug + 'static
{
}

macro_rules! native {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        // SAFETY: primitive integer/float types have no padding, accept every
        // bit pattern and have alignment <= 8, which divides 64.
        unsafe impl NativeType for $t {}
    )*};
}
native!(u8, i32, i64, u32, u64, f64);

/// Owner of a raw 64-byte aligned allocation.
struct RawBytes {
    ptr: NonNull<u8>,
    /// Allocated size in bytes (0 means "no allocation, dangling pointer").
    cap: usize,
}

// SAFETY: RawBytes uniquely owns its allocation; the bytes are plain data and
// are only mutated through `MutableBuffer` (which has unique ownership) before
// being frozen. Shared access after freezing is read-only.
unsafe impl Send for RawBytes {}
// SAFETY: see above; no interior mutability is exposed through `&RawBytes`.
unsafe impl Sync for RawBytes {}

impl RawBytes {
    fn dangling() -> Self {
        // Zero-length slices only need a non-null pointer aligned for T
        // (alignment <= 8), which `NonNull::<u64>::dangling()` provides.
        RawBytes {
            ptr: NonNull::<u64>::dangling().cast::<u8>(),
            cap: 0,
        }
    }

    fn layout(cap: usize) -> Result<Layout> {
        Layout::from_size_align(cap, ALIGNMENT).map_err(|_| ArrowError::Alloc(cap))
    }

    fn allocate(cap: usize, zeroed: bool) -> Result<Self> {
        if cap == 0 {
            return Ok(Self::dangling());
        }
        let layout = Self::layout(cap)?;
        // SAFETY: layout has non-zero size (cap > 0) and a valid alignment.
        let raw = unsafe {
            if zeroed {
                alloc::alloc_zeroed(layout)
            } else {
                alloc::alloc(layout)
            }
        };
        let ptr = NonNull::new(raw).ok_or(ArrowError::Alloc(cap))?;
        Ok(RawBytes { ptr, cap })
    }

    /// Grow to `new_cap` bytes (must be >= current cap); new bytes are zeroed.
    fn grow(&mut self, new_cap: usize) -> Result<()> {
        debug_assert!(new_cap >= self.cap);
        if self.cap == 0 {
            *self = Self::allocate(new_cap, true)?;
            return Ok(());
        }
        let old = Self::layout(self.cap)?;
        // Validate the new layout before reallocating.
        Self::layout(new_cap)?;
        // SAFETY: `self.ptr` was allocated by the global allocator with layout
        // `old`; `new_cap` is non-zero and does not overflow isize (checked by
        // Layout::from_size_align above).
        let raw = unsafe { alloc::realloc(self.ptr.as_ptr(), old, new_cap) };
        let ptr = NonNull::new(raw).ok_or(ArrowError::Alloc(new_cap))?;
        // SAFETY: the region [cap, new_cap) belongs to the new allocation.
        unsafe { std::ptr::write_bytes(ptr.as_ptr().add(self.cap), 0, new_cap - self.cap) };
        self.ptr = ptr;
        self.cap = new_cap;
        Ok(())
    }
}

impl Drop for RawBytes {
    fn drop(&mut self) {
        if self.cap > 0 {
            if let Ok(layout) = Self::layout(self.cap) {
                // SAFETY: allocated with exactly this layout by `allocate`/`grow`.
                unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
            }
        }
    }
}

fn padded(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(ALIGNMENT - 1)
        .map(|b| b / ALIGNMENT * ALIGNMENT)
        .ok_or(ArrowError::Alloc(bytes))
}

/// A growable, uniquely-owned, 64-byte aligned buffer of `T`.
pub struct MutableBuffer<T: NativeType> {
    raw: RawBytes,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: NativeType> Default for MutableBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativeType> MutableBuffer<T> {
    /// Empty buffer without allocation.
    pub fn new() -> Self {
        MutableBuffer {
            raw: RawBytes::dangling(),
            len: 0,
            _t: PhantomData,
        }
    }

    /// Empty buffer with room for at least `capacity` elements.
    ///
    /// Allocation failure degrades to an empty buffer that allocates lazily.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut b = Self::new();
        // Failure here is deferred to the first push, which reports it.
        let _ = b.try_reserve(capacity);
        b
    }

    /// A buffer of `len` zero-valued elements.
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(ArrowError::Alloc(usize::MAX))?;
        let raw = RawBytes::allocate(padded(bytes)?, true)?;
        Ok(MutableBuffer {
            raw,
            len,
            _t: PhantomData,
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity in elements.
    pub fn capacity(&self) -> usize {
        self.raw.cap / size_of::<T>()
    }

    /// Ensure room for `additional` more elements.
    pub fn try_reserve(&mut self, additional: usize) -> Result<()> {
        let needed = self
            .len
            .checked_add(additional)
            .ok_or(ArrowError::Alloc(usize::MAX))?;
        if needed <= self.capacity() {
            return Ok(());
        }
        let target = needed
            .max(self.capacity() * 2)
            .max(ALIGNMENT / size_of::<T>());
        let bytes = target
            .checked_mul(size_of::<T>())
            .ok_or(ArrowError::Alloc(usize::MAX))?;
        self.raw.grow(padded(bytes)?)
    }

    /// Append one element.
    ///
    /// On allocation failure the process aborts via `handle_alloc_error`,
    /// mirroring `Vec::push`.
    #[inline]
    pub fn push(&mut self, v: T) {
        if self.len == self.capacity() {
            if let Err(_e) = self.try_reserve(1) {
                alloc::handle_alloc_error(Layout::new::<T>());
            }
        }
        // SAFETY: len < capacity after reserve, so the slot is in bounds of the
        // allocation, and the pointer is aligned for T (64-byte base, T align <= 8).
        unsafe { self.raw.ptr.as_ptr().cast::<T>().add(self.len).write(v) };
        self.len += 1;
    }

    /// Append all elements of `s`.
    pub fn extend_from_slice(&mut self, s: &[T]) {
        if self.try_reserve(s.len()).is_err() {
            alloc::handle_alloc_error(Layout::new::<T>());
        }
        // SAFETY: capacity >= len + s.len(); source and destination cannot
        // overlap because `self` is uniquely borrowed mutably.
        unsafe {
            std::ptr::copy_nonoverlapping(
                s.as_ptr(),
                self.raw.ptr.as_ptr().cast::<T>().add(self.len),
                s.len(),
            )
        };
        self.len += s.len();
    }

    /// Append `n` copies of `v`.
    pub fn extend_constant(&mut self, n: usize, v: T) {
        if self.try_reserve(n).is_err() {
            alloc::handle_alloc_error(Layout::new::<T>());
        }
        let start = self.len;
        self.len += n;
        self.as_mut_slice()[start..].fill(v);
    }

    /// Shorten to `len` elements (no-op if already shorter).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// View as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` elements are initialised (written by push /
        // extend or zero-initialised), the pointer is aligned and non-null.
        unsafe { std::slice::from_raw_parts(self.raw.ptr.as_ptr().cast::<T>(), self.len) }
    }

    /// View as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, plus unique access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.raw.ptr.as_ptr().cast::<T>(), self.len) }
    }

    /// Freeze into an immutable shared [`Buffer`] without copying.
    pub fn into_buffer(self) -> Buffer<T> {
        let len = self.len;
        Buffer {
            data: Arc::new(self.raw),
            offset: 0,
            len,
            _t: PhantomData,
        }
    }
}

impl<T: NativeType> FromIterator<T> for MutableBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut b = MutableBuffer::with_capacity(iter.size_hint().0);
        for v in iter {
            b.push(v);
        }
        b
    }
}

/// An immutable, cheaply clonable, zero-copy sliceable buffer of `T`.
#[derive(Clone)]
pub struct Buffer<T: NativeType> {
    data: Arc<RawBytes>,
    /// Offset in elements into the allocation.
    offset: usize,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: NativeType> Buffer<T> {
    /// Empty buffer.
    pub fn empty() -> Self {
        MutableBuffer::new().into_buffer()
    }

    /// Copy a slice into a new aligned buffer.
    pub fn from_slice(s: &[T]) -> Self {
        let mut m = MutableBuffer::with_capacity(s.len());
        m.extend_from_slice(s);
        m.into_buffer()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `offset + len` elements lie within the initialised region of
        // the allocation (enforced by `slice`), the base pointer is 64-byte
        // aligned so `ptr + offset*size_of::<T>()` is aligned for T.
        unsafe {
            std::slice::from_raw_parts(
                self.data.ptr.as_ptr().cast::<T>().add(self.offset),
                self.len,
            )
        }
    }

    /// Zero-copy slice of `len` elements starting at `offset`.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| ArrowError::OutOfBounds("overflow".into()))?;
        if end > self.len {
            return Err(ArrowError::OutOfBounds(format!(
                "slice {offset}..{end} of buffer with len {}",
                self.len
            )));
        }
        Ok(Buffer {
            data: self.data.clone(),
            offset: self.offset + offset,
            len,
            _t: PhantomData,
        })
    }

    /// Raw pointer to the first element (for FFI export).
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Whether two buffers share the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

impl<T: NativeType> Default for Buffer<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: NativeType> From<Vec<T>> for Buffer<T> {
    fn from(v: Vec<T>) -> Self {
        Buffer::from_slice(&v)
    }
}

impl<T: NativeType> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<MutableBuffer<T>>().into_buffer()
    }
}

impl<T: NativeType> std::ops::Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: NativeType> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: NativeType> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_and_padded() {
        let mut b = MutableBuffer::<i64>::new();
        for i in 0..1000 {
            b.push(i);
        }
        assert_eq!(b.raw.ptr.as_ptr() as usize % ALIGNMENT, 0);
        assert_eq!(b.raw.cap % ALIGNMENT, 0);
        let buf = b.into_buffer();
        assert_eq!(buf.len(), 1000);
        assert_eq!(buf[999], 999);
    }

    #[test]
    fn slicing_is_zero_copy() {
        let buf: Buffer<i32> = (0..100).collect();
        let s = buf.slice(10, 20).unwrap();
        assert!(s.ptr_eq(&buf));
        assert_eq!(s.as_slice(), &(10..30).collect::<Vec<_>>()[..]);
        let s2 = s.slice(5, 5).unwrap();
        assert_eq!(s2.as_slice(), &[15, 16, 17, 18, 19]);
        assert!(buf.slice(95, 10).is_err());
    }

    #[test]
    fn empty_and_zeroed() {
        let e = Buffer::<f64>::empty();
        assert!(e.is_empty());
        assert_eq!(e.as_slice(), &[] as &[f64]);
        let z = MutableBuffer::<u64>::zeroed(17).unwrap();
        assert!(z.as_slice().iter().all(|&x| x == 0));
        let mut c = MutableBuffer::<u8>::new();
        c.extend_constant(130, 7);
        c.extend_from_slice(&[1, 2, 3]);
        c.truncate(131);
        assert_eq!(c.len(), 131);
        assert_eq!(c.as_slice()[130], 1);
    }

    #[test]
    fn shared_across_threads() {
        let buf: Buffer<i64> = (0..1000).collect();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let b = buf.slice(i * 100, 100).unwrap();
                std::thread::spawn(move || b.iter().sum::<i64>())
            })
            .collect();
        let total: i64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, (0..400).sum::<i64>());
    }
}
