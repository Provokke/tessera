//! Small helpers shared by the kernels.
use tessera_arrow::Bitmap;

use crate::error::Result;

/// Validity of a binary kernel: AND of the inputs' validity (null if either is).
pub(crate) fn and_validity(a: Option<&Bitmap>, b: Option<&Bitmap>) -> Result<Option<Bitmap>> {
    Ok(match (a, b) {
        (Some(p), Some(q)) => Some(p.and(q)?),
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (None, None) => None,
    })
}
