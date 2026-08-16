import tessera


def test_from_dict_shape_select_to_dict():
    df = tessera.DataFrame({"a": [1, None, 3], "b": ["x", "y", None], "c": [0.5, 1.5, None]})
    assert df.shape == (3, 3)
    assert df.select(["c", "a"]).to_dict() == {"c": [0.5, 1.5, None], "a": [1, None, 3]}


def test_filter_matches_python_reference():
    a = [5, None, -2, 7, 0, 3]
    df = tessera.DataFrame({"a": a, "i": list(range(len(a)))})
    out = df.filter("a", ">", 0).to_dict()
    assert out["a"] == [x for x in a if x is not None and x > 0]
    assert out["i"] == [i for i, x in enumerate(a) if x is not None and x > 0]
