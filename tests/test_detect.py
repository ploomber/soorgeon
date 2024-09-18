import testutils

import pytest
import parso

from soorgeon import detect


@pytest.mark.parametrize(
    "code, expected",
    [
        ["x = 1", True],
        ["x, y = 1, 2", True],
        ["y, x = 1, 2", True],
        ["(x, y) = 1, 2", True],
        ["(y, x) = 1, 2", True],
        ["[x, y] = 1, 2", True],
        ["[y, x] = 1, 2", True],
        ["(z, (y, x)) = 1, (2, 3)", True],
        ["x(1)", False],
        ["something(x)", False],
        ["x == 1", False],
    ],
)
def test_is_left_side_of_assignment(code, expected):
    node = testutils.get_first_leaf_with_value(code, "x")
    assert detect.is_left_side_of_assignment(node) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["for x in range(10):\n    pass", True],
        ["for x, y in range(10):\n    pass", True],
        ["for y, (z, x) in range(10):\n    pass", True],
        ["for y in range(10):\n    x = y + 1", False],
        ["for y in range(10):\n    z = y + 1\nfunction(x)", False],
    ],
    ids=[
        "single",
        "tuple",
        "nested",
        "variable-in-loop-body",
        "variable-in-loop-body-2",
    ],
)
def test_is_for_loop(code, expected):
    leaf = testutils.get_first_leaf_with_value(code, "x")
    assert detect.is_for_loop(leaf) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["def a():\n    pass", False],
        ["class A:\n    pass", True],
        ["class A:\n    def __init__(self):\n        pass", True],
    ],
    ids=[
        "function",
        "class-empty",
        "class",
    ],
)
def test_is_classdef(code, expected):
    leaf = parso.parse(code).get_first_leaf()
    assert detect.is_classdef(leaf) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["[1, 2, 3]", False],
        ["{1, 2, 3}", False],
        ["{1: 1, 2: 2, 3: 3}", False],
        ["[x for x in range(10)]", True],
        ["{x for x in range(10)}", True],
        ["{x for x in range(10) if x > 1}", True],
        ["{x: x + 1 for x in range(10)}", True],
        ["{x: x + 1 for x in range(10) if x > 1 and x < 8}", True],
        ["(x for x in range(10))", True],
    ],
    ids=[
        "list",
        "set",
        "dict",
        "simple",
        "set-comp",
        "set-comp-conditional",
        "dict-comp",
        "dict-comp-conditional",
        "generator",
    ],
)
def test_is_comprehension(code, expected):
    leaf = parso.parse(code).get_first_leaf()
    assert detect.is_comprehension(leaf) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["for x in range(10):\n    pass", False],
        ["with open(x) as f:\n    pass", True],
    ],
    ids=[
        "not-context-manager",
        "simple",
    ],
)
def test_is_context_manager(code, expected):
    leaf = testutils.get_first_leaf_with_value(code, "x")
    assert detect.is_context_manager(leaf) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["[x for x in range(10)]", True],
        ["[x, y]", False],
        ["[x.attribute for x in range(10)", True],
        ["[x for x in range(10) if x > 0", True],
    ],
    ids=[
        "for",
        "list",
        "attribute",
        "conditional",
    ],
)
def test_is_inside_list_comprehension(code, expected):
    node = testutils.get_first_leaf_with_value(code, "x")
    assert detect.is_inside_list_comprehension(node) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["sns.histplot(df.some_column)", True],
        ["histplot(df.some_column)", True],
        ["sns.histplot(df)", True],
        ["histplot(df)", True],
        ['sns.histplot(df["key"])', True],
        ["def x(df):\n  pass", False],
        ["def x(df=1):\n  pass", False],
        ["(df, df2) = 1, 2", False],
        ['function({"data": df})', True],
        ["function(dict(data=df))", True],
        ['function({"data": (df - 1)})', True],
        ['Constructor({"data": df}).do_stuff()', True],
        ['Constructor({"data": (df - 1)}).do_stuff()', True],
    ],
    ids=[
        "arg-attribute",
        "arg-attribute-2",
        "arg",
        "arg-2",
        "arg-getitem",
        "fn-signature",
        "fn-signature-default-value",
        "assignment",
        "arg-nested-dict",
        "arg-nested-dict-constructor",
        "arg-nested-dict-operation",
        "constructor-dict",
        "constructor-dict-operation",
    ],
)
def test_inside_function_call(code, expected):
    leaf = testutils.get_first_leaf_with_value(code, "df")
    assert detect.is_inside_function_call(leaf) is expected
