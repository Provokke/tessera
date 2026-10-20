//! Typed arrays following the Arrow physical layout.
use crate::bitmap::Bitmap;
use crate::buffer::{Buffer, NativeType};
use crate::datatype::{DataType, Scalar};
use crate::error::{ArrowError, Result};

fn check_validity(validity: &Option<Bitmap>, len: usize) -> Result<()> {
    match validity {
        Some(v) if v.len() != len => Err(ArrowError::LengthMismatch(format!(
            "validity has {} bits, array has {len} values",
            v.len()
        ))),
        _ => Ok(()),
    }
}

fn slice_validity(v: &Option<Bitmap>, offset: usize, len: usize) -> Result<Option<Bitmap>> {
    v.as_ref().map(|b| b.slice(offset, len)).transpose()
}

/// A fixed-width primitive array with optional validity bitmap.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveArray<T: NativeType> {
    values: Buffer<T>,
    validity: Option<Bitmap>,
}

impl<T: NativeType> PrimitiveArray<T> {
    /// Construct, validating that the validity length matches.
    pub fn try_new(values: Buffer<T>, validity: Option<Bitmap>) -> Result<Self> {
        check_validity(&validity, values.len())?;
        Ok(PrimitiveArray { values, validity })
    }
    /// Array with no nulls.
    pub fn from_values(values: Buffer<T>) -> Self {
        PrimitiveArray {
            values,
            validity: None,
        }
    }
    /// From optional values (null slots are stored as `T::default()`).
    pub fn from_options<I: IntoIterator<Item = Option<T>>>(it: I) -> Self {
        let mut b = crate::builder::PrimitiveBuilder::new();
        for v in it {
            b.push(v);
        }
        b.finish()
    }
    /// Number of slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Value buffer (slots behind nulls are unspecified).
    pub fn values(&self) -> &Buffer<T> {
        &self.values
    }
    /// Validity bitmap, if any.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }
    /// Number of nulls.
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::unset_bits)
    }
    /// Whether slot `i` is valid.
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity.as_ref().is_none_or(|v| v.get(i))
    }
    /// Value at `i`, or `None` when null or out of bounds.
    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.len() && self.is_valid(i) {
            Some(self.values[i])
        } else {
            None
        }
    }
    /// Iterate optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<T>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
    /// Zero-copy slice.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        Ok(PrimitiveArray {
            values: self.values.slice(offset, len)?,
            validity: slice_validity(&self.validity, offset, len)?,
        })
    }
}

/// A boolean array: values and validity are both bitmaps.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanArray {
    values: Bitmap,
    validity: Option<Bitmap>,
}

impl BooleanArray {
    /// Construct, validating lengths.
    pub fn try_new(values: Bitmap, validity: Option<Bitmap>) -> Result<Self> {
        check_validity(&validity, values.len())?;
        Ok(BooleanArray { values, validity })
    }
    /// From optional bools.
    pub fn from_options<I: IntoIterator<Item = Option<bool>>>(it: I) -> Self {
        let mut b = crate::builder::BooleanBuilder::new();
        for v in it {
            b.push(v);
        }
        b.finish()
    }
    /// Number of slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Value bitmap.
    pub fn values(&self) -> &Bitmap {
        &self.values
    }
    /// Validity bitmap.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }
    /// Number of nulls.
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::unset_bits)
    }
    /// Value at `i`, `None` if null/out of bounds.
    pub fn get(&self, i: usize) -> Option<bool> {
        if i < self.len() && self.validity.as_ref().is_none_or(|v| v.get(i)) {
            Some(self.values.get(i))
        } else {
            None
        }
    }
    /// Iterate optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<bool>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
    /// Zero-copy slice.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        Ok(BooleanArray {
            values: self.values.slice(offset, len)?,
            validity: slice_validity(&self.validity, offset, len)?,
        })
    }
}

/// Variable-length UTF-8 strings: `i32` offsets (len + 1) and a data buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Utf8Array {
    offsets: Buffer<i32>,
    data: Buffer<u8>,
    validity: Option<Bitmap>,
}

impl Utf8Array {
    /// Construct, validating offsets (non-empty, non-negative, monotone,
    /// within `data`), UTF-8 of every value, and validity length.
    pub fn try_new(
        offsets: Buffer<i32>,
        data: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Result<Self> {
        let first = *offsets
            .first()
            .ok_or_else(|| ArrowError::InvalidOffsets("empty offsets".into()))?;
        if first < 0 {
            return Err(ArrowError::InvalidOffsets("negative offset".into()));
        }
        if offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err(ArrowError::InvalidOffsets("offsets decrease".into()));
        }
        let last = offsets[offsets.len() - 1] as usize;
        if last > data.len() {
            return Err(ArrowError::InvalidOffsets(format!(
                "last offset {last} > data len {}",
                data.len()
            )));
        }
        for w in offsets.windows(2) {
            std::str::from_utf8(&data[w[0] as usize..w[1] as usize])
                .map_err(|_| ArrowError::InvalidUtf8)?;
        }
        check_validity(&validity, offsets.len() - 1)?;
        Ok(Utf8Array {
            offsets,
            data,
            validity,
        })
    }
    /// From optional strings.
    pub fn from_options<'a, I: IntoIterator<Item = Option<&'a str>>>(it: I) -> Result<Self> {
        let mut b = crate::builder::Utf8Builder::new();
        for v in it {
            b.push(v);
        }
        b.finish()
    }
    /// Number of slots.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Offsets buffer.
    pub fn offsets(&self) -> &Buffer<i32> {
        &self.offsets
    }
    /// Data buffer.
    pub fn data(&self) -> &Buffer<u8> {
        &self.data
    }
    /// Validity bitmap.
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }
    /// Number of nulls.
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::unset_bits)
    }
    /// String at `i` ignoring validity. Panics if `i >= len`.
    pub fn value(&self, i: usize) -> &str {
        let (s, e) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        // SAFETY: `try_new` validated that every [offsets[i], offsets[i+1])
        // range is in bounds and valid UTF-8, and slicing preserves ranges.
        unsafe { std::str::from_utf8_unchecked(&self.data[s..e]) }
    }
    /// String at `i`, `None` if null/out of bounds.
    pub fn get(&self, i: usize) -> Option<&str> {
        if i < self.len() && self.validity.as_ref().is_none_or(|v| v.get(i)) {
            Some(self.value(i))
        } else {
            None
        }
    }
    /// Iterate optional strings.
    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
    /// Zero-copy slice (offsets are sliced; data is shared, not rebased).
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        Ok(Utf8Array {
            offsets: self.offsets.slice(offset, len + 1)?,
            data: self.data.clone(),
            validity: slice_validity(&self.validity, offset, len)?,
        })
    }
}

/// A type-erased array.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    /// Booleans.
    Boolean(BooleanArray),
    /// Int32.
    Int32(PrimitiveArray<i32>),
    /// Int64.
    Int64(PrimitiveArray<i64>),
    /// Float64.
    Float64(PrimitiveArray<f64>),
    /// Utf8.
    Utf8(Utf8Array),
}

impl Array {
    /// Logical type.
    pub fn data_type(&self) -> DataType {
        match self {
            Array::Boolean(_) => DataType::Boolean,
            Array::Int32(_) => DataType::Int32,
            Array::Int64(_) => DataType::Int64,
            Array::Float64(_) => DataType::Float64,
            Array::Utf8(_) => DataType::Utf8,
        }
    }
    /// Length.
    pub fn len(&self) -> usize {
        match self {
            Array::Boolean(a) => a.len(),
            Array::Int32(a) => a.len(),
            Array::Int64(a) => a.len(),
            Array::Float64(a) => a.len(),
            Array::Utf8(a) => a.len(),
        }
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Null count.
    pub fn null_count(&self) -> usize {
        match self {
            Array::Boolean(a) => a.null_count(),
            Array::Int32(a) => a.null_count(),
            Array::Int64(a) => a.null_count(),
            Array::Float64(a) => a.null_count(),
            Array::Utf8(a) => a.null_count(),
        }
    }
    /// Value at `i` as a [`Scalar`].
    pub fn scalar(&self, i: usize) -> Scalar {
        let dt = self.data_type();
        let s = match self {
            Array::Boolean(a) => a.get(i).map(Scalar::Boolean),
            Array::Int32(a) => a.get(i).map(Scalar::Int32),
            Array::Int64(a) => a.get(i).map(Scalar::Int64),
            Array::Float64(a) => a.get(i).map(Scalar::Float64),
            Array::Utf8(a) => a.get(i).map(|s| Scalar::Utf8(s.to_owned())),
        };
        s.unwrap_or(Scalar::Null(dt))
    }
    /// Zero-copy slice.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        Ok(match self {
            Array::Boolean(a) => Array::Boolean(a.slice(offset, len)?),
            Array::Int32(a) => Array::Int32(a.slice(offset, len)?),
            Array::Int64(a) => Array::Int64(a.slice(offset, len)?),
            Array::Float64(a) => Array::Float64(a.slice(offset, len)?),
            Array::Utf8(a) => Array::Utf8(a.slice(offset, len)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_nulls_and_slice() {
        let a = PrimitiveArray::from_options([Some(1i64), None, Some(3), None, Some(5)]);
        assert_eq!(a.null_count(), 2);
        let s = a.slice(1, 3).unwrap();
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![None, Some(3), None]);
        assert!(s.values().ptr_eq(a.values()));
        assert_eq!(s.null_count(), 2);
        assert!(a.slice(4, 2).is_err());
    }

    #[test]
    fn utf8_roundtrip_and_slice() {
        let a = Utf8Array::from_options([Some("héllo"), None, Some(""), Some("wörld")]).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(a.get(0), Some("héllo"));
        assert_eq!(a.get(1), None);
        assert_eq!(a.get(2), Some(""));
        let s = a.slice(2, 2).unwrap();
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![Some(""), Some("wörld")]);
        assert!(s.data().ptr_eq(a.data()));
    }

    #[test]
    fn utf8_validation() {
        let bad = Utf8Array::try_new(vec![0, 3, 2].into(), b"abc".to_vec().into(), None);
        assert!(matches!(bad, Err(ArrowError::InvalidOffsets(_))));
        let bad = Utf8Array::try_new(vec![0, 4].into(), b"abc".to_vec().into(), None);
        assert!(bad.is_err());
        let bad = Utf8Array::try_new(vec![0, 2].into(), vec![0xC3u8, 0x28].into(), None);
        assert_eq!(bad, Err(ArrowError::InvalidUtf8));
        let bad = Utf8Array::try_new(Buffer::empty(), Buffer::empty(), None);
        assert!(bad.is_err());
    }

    #[test]
    fn validity_length_checked() {
        let v = Bitmap::from_bools([true, false]);
        assert!(PrimitiveArray::<i32>::try_new(vec![1, 2, 3].into(), Some(v)).is_err());
    }

    #[test]
    fn boolean_and_scalar() {
        let a = Array::Boolean(BooleanArray::from_options([Some(true), None, Some(false)]));
        assert_eq!(a.scalar(0), Scalar::Boolean(true));
        assert_eq!(a.scalar(1), Scalar::Null(DataType::Boolean));
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.slice(1, 2).unwrap().len(), 2);
    }
}
