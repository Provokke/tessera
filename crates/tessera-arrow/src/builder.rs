//! Builders that append optional values and freeze into arrays.
use crate::array::{BooleanArray, PrimitiveArray, Utf8Array};
use crate::bitmap::MutableBitmap;
use crate::buffer::{MutableBuffer, NativeType};
use crate::error::{ArrowError, Result};

/// Builder for [`PrimitiveArray`]. The validity bitmap is only materialized
/// once the first null is pushed.
pub struct PrimitiveBuilder<T: NativeType> {
    values: MutableBuffer<T>,
    validity: Option<MutableBitmap>,
}

impl<T: NativeType> Default for PrimitiveBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativeType> PrimitiveBuilder<T> {
    /// New empty builder.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    /// With capacity.
    pub fn with_capacity(n: usize) -> Self {
        PrimitiveBuilder {
            values: MutableBuffer::with_capacity(n),
            validity: None,
        }
    }
    /// Append a value or null.
    #[inline]
    pub fn push(&mut self, v: Option<T>) {
        match v {
            Some(v) => {
                self.values.push(v);
                if let Some(b) = &mut self.validity {
                    b.push(true);
                }
            }
            None => {
                if self.validity.is_none() {
                    let mut b = MutableBitmap::with_capacity(self.values.capacity());
                    b.extend_constant(self.values.len(), true);
                    self.validity = Some(b);
                }
                self.values.push(T::default());
                if let Some(b) = &mut self.validity {
                    b.push(false);
                }
            }
        }
    }
    /// Number of appended slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Freeze.
    pub fn finish(self) -> PrimitiveArray<T> {
        // Lengths match by construction, so this cannot fail; fall back to
        // dropping validity rather than panicking.
        let values = self.values.into_buffer();
        let validity = self.validity.map(MutableBitmap::into_bitmap);
        match PrimitiveArray::try_new(values.clone(), validity) {
            Ok(a) => a,
            Err(_) => PrimitiveArray::from_values(values),
        }
    }
}

/// Builder for [`BooleanArray`].
#[derive(Default)]
pub struct BooleanBuilder {
    values: MutableBitmap,
    validity: Option<MutableBitmap>,
}

impl BooleanBuilder {
    /// New builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append.
    pub fn push(&mut self, v: Option<bool>) {
        if v.is_none() && self.validity.is_none() {
            let mut b = MutableBitmap::new();
            b.extend_constant(self.values.len(), true);
            self.validity = Some(b);
        }
        self.values.push(v.unwrap_or(false));
        if let Some(b) = &mut self.validity {
            b.push(v.is_some());
        }
    }
    /// Freeze.
    pub fn finish(self) -> BooleanArray {
        let values = self.values.into_bitmap();
        let validity = self.validity.map(MutableBitmap::into_bitmap);
        match BooleanArray::try_new(values.clone(), validity) {
            Ok(a) => a,
            Err(_) => BooleanArray::from_options(values.iter().map(Some)),
        }
    }
}

/// Builder for [`Utf8Array`]; `finish` errors if data exceeds `i32::MAX` bytes.
pub struct Utf8Builder {
    offsets: MutableBuffer<i32>,
    data: MutableBuffer<u8>,
    validity: Option<MutableBitmap>,
    overflow: bool,
}

impl Default for Utf8Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Utf8Builder {
    /// New builder.
    pub fn new() -> Self {
        let mut offsets = MutableBuffer::new();
        offsets.push(0);
        Utf8Builder {
            offsets,
            data: MutableBuffer::new(),
            validity: None,
            overflow: false,
        }
    }
    /// Append.
    pub fn push(&mut self, v: Option<&str>) {
        let n = self.offsets.len() - 1;
        if v.is_none() && self.validity.is_none() {
            let mut b = MutableBitmap::new();
            b.extend_constant(n, true);
            self.validity = Some(b);
        }
        if let Some(s) = v {
            self.data.extend_from_slice(s.as_bytes());
        }
        match i32::try_from(self.data.len()) {
            Ok(o) => self.offsets.push(o),
            Err(_) => {
                self.overflow = true;
                self.offsets.push(i32::MAX);
            }
        }
        if let Some(b) = &mut self.validity {
            b.push(v.is_some());
        }
    }
    /// Freeze.
    pub fn finish(self) -> Result<Utf8Array> {
        if self.overflow {
            return Err(ArrowError::OffsetOverflow);
        }
        Utf8Array::try_new(
            self.offsets.into_buffer(),
            self.data.into_buffer(),
            self.validity.map(MutableBitmap::into_bitmap),
        )
    }
}
