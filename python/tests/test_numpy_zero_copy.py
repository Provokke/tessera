"""Zero-copy NumPy export through the buffer protocol."""
import gc

import numpy as np
import pytest

import tessera


def test_int64_view_shares_rust_memory():
    df = tessera.DataFrame({"v": list(range(1000)), "s": [str(i) for i in range(1000)]})
    view = df.column_view("v")
    arr = np.asarray(view)
    assert arr.dtype == np.int64 and arr.shape == (1000,)
    assert arr.ctypes.data == view.data_ptr()  # the numpy array points at the Rust buffer
    assert arr.tolist() == list(range(1000))
    assert not arr.flags.writeable
    with pytest.raises((ValueError, TypeError)):
        arr[0] = 1
    # a second export and a memoryview see the same address
    assert np.asarray(view).ctypes.data == view.data_ptr()
    mv = memoryview(view)
    assert mv.readonly and mv.format == "q" and len(mv) == 1000


def test_float_and_pointer_stable_through_select_and_head():
    df = tessera.DataFrame({"f": [0.5 * i for i in range(100)], "i": list(range(100))})
    base = df.column_view("f").data_ptr()
    assert df.select(["f"]).column_view("f").data_ptr() == base  # select is zero-copy
    assert df.head(10).column_view("f").data_ptr() == base  # head is a zero-copy slice
    a = np.asarray(df.head(10).column_view("f"))
    assert a.dtype == np.float64 and a.shape == (10,)
    assert a.ctypes.data == base
    assert np.shares_memory(a, np.asarray(df.column_view("f")))
    np.testing.assert_array_equal(a, np.arange(10) * 0.5)


def test_array_outlives_dataframe():
    df = tessera.DataFrame({"v": list(range(100_000))})
    arr = np.asarray(df.column_view("v"))
    del df
    gc.collect()
    assert int(arr.sum()) == sum(range(100_000))


def test_nulls_and_unsupported_types():
    df = tessera.DataFrame({"v": [1, None, 3], "s": ["a", "b", "c"], "b": [True, False, True]})
    assert df.column_view("v").null_count == 1
    assert len(df.column_view("v")) == 3
    for c in ("s", "b"):
        with pytest.raises(TypeError):
            df.column_view(c)
    with pytest.raises(ValueError):
        df.column_view("missing")
