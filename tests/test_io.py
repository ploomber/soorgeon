import testutils
from conftest import read_snippets

import parso
import pytest

from soorgeon import io


def get_first_sibling_after_assignment(code, index):
    tree = parso.parse(code)
    leaf = tree.get_first_leaf()
    current = 0

    while leaf:
        if leaf.value == "=":
            if index == current:
                break

            current += 1

        leaf = leaf.get_next_leaf()

    return leaf.get_next_sibling()


@pytest.mark.parametrize(
    "code, expected_len",
    [
        ["[x for x in range(10)]", 1],
        ["[i for row in matrix for i in row]", 2],
        ["[i for matrix in tensor for row in matrix for i in row]", 3],
    ],
    ids=[
        "simple",
        "nested",
        "nested-nested",
    ],
)
def test_flatten_sync_comp_for(code, expected_len):
    synccompfor = parso.parse(code).children[0].children[1].children[1]

    assert len(io._flatten_sync_comp_for(synccompfor)) == expected_len


@pytest.mark.parametrize(
    "code, in_expected, declared_expected",
    [
        ["[x for x in range(10)]", set(), {"x"}],
    ],
    ids=[
        "simple",
    ],
)
def test_find_sync_comp_for_inputs_and_scope(code, in_expected, declared_expected):
    synccompfor = parso.parse(code).children[0].children[1].children[1]

    in_, declared = io._find_sync_comp_for_inputs_and_scope(synccompfor)

    assert in_ == in_expected
    assert declared == declared_expected


@pytest.mark.parametrize(
    "snippets, local_scope, expected",
    [
        ['import pandas as pd\n df = pd.read_csv("data.csv")', {"pd"}, (set(), {"df"})],
        [
            "import pandas as pd\nimport do_stuff\n"
            'df = do_stuff(pd.read_csv("data.csv"))',
            {"pd"},
            (set(), {"df"}),
        ],
    ],
    ids=["simple", "inside-function"],
)
def test_find_inputs_and_outputs_local_scope(snippets, local_scope, expected):
    assert io.find_inputs_and_outputs(snippets, local_scope) == expected


first = """
x = 1
"""

second = """
y = x + 1
"""

# NOTE: if we change this to x = y + 1, this will break
# we need to expect name clashes, and should prioritize the ones that
# appear first in the notebook
third = """
z = y + 1
"""

# exploratory data analysis example
eda = {
    "load": "import load_data, plot\ndf = load_data()",
    # note that we re-define df
    "clean": "df = df[df.some_columns > 2]",
    "plot": "plot(df)",
}

# imports on its own section
imports = {
    "imports": "import pandas as pd",
    "load": 'df = pd.read_csv("data.csv")',
}


def test_providermapping():
    m = io.ProviderMapping(io.find_io(eda))

    assert m._providers_for_task("load") == {}
    assert m._providers_for_task("clean") == {"df": "load"}
    assert m._providers_for_task("plot") == {"df": "clean"}
    assert m.get("df", "clean") == "load"


def test_providermapping_error():
    m = io.ProviderMapping(io.find_io(eda))

    with pytest.raises(KeyError) as excinfo:
        m.get("unknown_variable", "clean")

    expected = (
        "\"Error parsing inputs for section 'clean' notebook: "
        "could not find an earlier section declaring "
        "variable 'unknown_variable'\""
    )

    assert expected == str(excinfo.value)


@pytest.mark.parametrize(
    "snippets, expected",
    [
        [
            {"first": first, "second": second, "third": third},
            {"first": [], "second": ["first"], "third": ["second"]},
        ],
        [eda, {"load": [], "clean": ["load"], "plot": ["clean"]}],
        [
            read_snippets("ml"),
            {
                "load": [],
                "clean": ["load"],
                "train-test-split": ["clean"],
                "linear-regression": ["train-test-split"],
                "random-forest-regressor": ["train-test-split"],
            },
        ],
    ],
)
def test_find_upstream(snippets, expected):
    assert io.find_upstream(snippets) == expected


@pytest.mark.parametrize(
    "snippets, expected",
    [
        [
            eda,
            {
                "load": (set(), {"df"}),
                "clean": ({"df"}, {"df"}),
                "plot": ({"df"}, set()),
            },
        ],
        [
            read_snippets("ml"),
            {
                "load": (set(), {"df", "ca_housing"}),
                "clean": ({"df"}, {"df"}),
                "train-test-split": (
                    {"df"},
                    {"y", "X", "X_train", "X_test", "y_train", "y_test"},
                ),
                "linear-regression": (
                    {"y_test", "X_test", "y_train", "X_train"},
                    {"lr", "y_pred"},
                ),
                "random-forest-regressor": (
                    {"y_test", "X_test", "y_train", "X_train"},
                    {"rf", "y_pred"},
                ),
            },
        ],
        [imports, {"imports": (set(), set()), "load": (set(), {"df"})}],
    ],
    ids=[
        "eda",
        "ml",
        "imports",
    ],
)
def test_find_io(snippets, expected):
    assert io.find_io(snippets) == expected


@pytest.mark.parametrize(
    "io_, expected",
    [
        [
            {
                "one": ({"a"}, {"b", "c"}),
                "two": ({"b"}, set()),
            },
            {
                "one": ({"a"}, {"b"}),
                "two": ({"b"}, set()),
            },
        ],
    ],
)
def test_prune_io(io_, expected):
    assert io.prune_io(io_) == expected


exploratory = """
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris(as_frame=True)['data']

df = df[df['petal length (cm)'] > 2]

sns.histplot(df['petal length (cm)'])
"""


@pytest.mark.parametrize(
    "code_nb, code_task, expected",
    [
        [
            exploratory,
            "df = load_iris(as_frame=True)['data']",
            "from sklearn.datasets import load_iris",
        ],
        [
            exploratory,
            "df = df[df['petal length (cm)'] > 2]",
            None,
        ],
        [
            exploratory,
            "sns.histplot(df['petal length (cm)'])",
            "import seaborn as sns",
        ],
    ],
)
def test_importsparser(code_nb, code_task, expected):
    ip = io.ImportsParser(code_nb)
    assert ip.get_imports_cell_for_task(code_task) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["import pandas as pd\nimport numpy as np", "\n"],
        ["import math", ""],
        ["import pandas as pd\n1+1", "\n1+1"],
        ["import math\n1+1", "\n1+1"],
    ],
)
def test_remove_imports(code, expected):
    assert io.remove_imports(code) == expected


@pytest.mark.parametrize(
    "snippets, names, a, b",
    [
        [
            {"a": "import pandas", "b": "import numpy as np"},
            {"a": {"pandas"}, "b": {"np"}},
            set(),
            {"pandas"},
        ],
        [
            {
                "a": "def x():\n    pass",
                "b": "def y():\n    pass",
            },
            {"a": {"x"}, "b": {"y"}},
            set(),
            {"x"},
        ],
    ],
)
def test_definitions_mapping(snippets, names, a, b):
    im = io.DefinitionsMapping(snippets)

    assert im._names == names
    assert im.get("a") == a
    assert im.get("b") == b


@pytest.mark.parametrize(
    "code, def_expected, in_expected, out_expected",
    [
        ["for x in range(10):\n    pass", {"x"}, set(), set()],
        ["for x, y in range(10):\n    pass", {"x", "y"}, set(), set()],
        ["for x, (y, z) in range(10):\n    pass", {"x", "y", "z"}, set(), set()],
        ["for x in range(10):\n    pass\n\nj = i", {"x"}, set(), set()],
        [
            "for i, a_range in enumerate(range(x)):\n    pass",
            {"i", "a_range"},
            {"x"},
            set(),
        ],
        ["for i in range(10):\n    print(i + 10)", {"i"}, set(), set()],
    ],
    ids=[
        "one",
        "two",
        "nested",
        "code-outside-for-loop",
        "nested-calls",
        "uses-local-sope-in-body",
    ],
)
def test_find_for_loop_def_and_io(code, def_expected, in_expected, out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = io.find_for_loop_def_and_io(tree.children[0])
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize(
    "code, def_expected, in_expected, out_expected",
    [
        ['with open("file") as f:\n    pass', {"f"}, set(), set()],
        ['with open("file"):\n    pass', set(), set(), set()],
        [
            'with open("file") as f, open("another") as g:\n    pass',
            {"f", "g"},
            set(),
            set(),
        ],
        ['with open("file") as f:\n    x = f.read()', {"f"}, set(), {"x"}],
        ['with open("file") as f:\n    x, y = f.read()', {"f"}, set(), {"x", "y"}],
        ["with open(some_path) as f:\n    x = f.read()", {"f"}, {"some_path"}, {"x"}],
        [
            "with open(some_path, another_path) as (f, ff):\n    x = f.read()",
            {"f", "ff"},
            {"some_path", "another_path"},
            {"x"},
        ],
    ],
    ids=[
        "one",
        "no-alias",
        "two",
        "output-one",
        "output-many",
        "input-one",
        "input-many",
    ],
)
def test_find_context_manager_def_and_io(code, def_expected, in_expected, out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = io.find_context_manager_def_and_io(tree.children[0])
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize(
    "code, def_expected, in_expected, out_expected",
    [
        ["def fn(x):\n    pass", {"x"}, set(), set()],
        ["def fn(x, y):\n    pass", {"x", "y"}, set(), set()],
        ["def fn(x, y):\n    something = z + x + y", {"x", "y"}, {"z"}, set()],
        ["def fn(x, y):\n    z(x, y)", {"x", "y"}, {"z"}, set()],
        ["def fn(x, y):\n    z.do(x, y)", {"x", "y"}, {"z"}, set()],
        ["def fn(x, y):\n    z[x]", {"x", "y"}, {"z"}, set()],
        ["def fn(x, y):\n    z + x + y", {"x", "y"}, {"z"}, set()],
        ["def fn(x, y):\n    z", {"x", "y"}, {"z"}, set()],
        ["def fn() -> Mapping[str, int]:\n    pass", set(), {"Mapping"}, set()],
        [
            "def fn(x: int, y: Mapping[str, int]):\n    z + x + y",
            {"Mapping", "x", "y"},
            {"z"},
            set(),
        ],
        [
            "def fn(a=1):\n    pass",
            {"a"},
            set(),
            set(),
        ],
        [
            "def fn(a: str=1):\n    pass",
            {"a"},
            set(),
            set(),
        ],
    ],
    ids=[
        "arg-one",
        "arg-two",
        "uses-outer-scope",
        "uses-outer-scope-callable",
        "uses-outer-scope-attribute",
        "uses-outer-scope-getitem",
        "uses-outer-scope-no-assignment",
        "uses-outer-scope-reference",
        "annotation-return",
        "annotation-args",
        "kwargs",
        "annotation-kwargs",
    ],
)
def test_find_function_scope_and_io(code, def_expected, in_expected, out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = io.find_function_scope_and_io(tree.children[0])
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["name(x, y)", {"name", "x", "y"}],
        ["name(a=x, b=y)", {"name", "x", "y"}],
        ["name(x, b=y)", {"name", "x", "y"}],
        ['name({"x": x}, b=y)', {"name", "x", "y"}],
        ['name(x, b={"y": y})', {"name", "x", "y"}],
        ["name([x, y])", {"name", "x", "y"}],
        ['name["a"]', {"name"}],
        ["name.atribute", {"name"}],
    ],
    ids=[
        "simple",
        "keywords",
        "mixed",
        "arg-dict",
        "keyarg-dict",
        "arg-list",
        "getitem",
        "attribute",
    ],
)
def test_find_inputs(code, expected):
    atom_exp = testutils.get_first_leaf_with_value(code, "name").parent
    assert io.find_inputs(atom_exp) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ['name["a"]', {"name"}],
        ["name.atribute", {"name"}],
        ["name", set()],
    ],
    ids=[
        "getitem",
        "attribute",
        "name",
    ],
)
def test_find_inputs_only_getitem_and_attribute_access(code, expected):
    atom_exp = testutils.get_first_leaf_with_value(code, "name").parent
    out = io.find_inputs(atom_exp, only_getitem_and_attribute_access=True)
    assert out == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ['[x for x in df["some_key"]]', {"df"}],
        ['[x for x in df["some_key"]["another_key"]]', {"df"}],
    ],
    ids=[
        "getitem",
        "getitem-nested",
    ],
)
def test_find_inputs_only_getitem_and_attribute_access_list_comprehension(
    code, expected
):
    out = io.find_inputs(parso.parse(code), only_getitem_and_attribute_access=True)
    assert out == expected


@pytest.mark.parametrize(
    "code, expected, index",
    [
        ["x = df.something", {"df"}, 0],
        ["x = df.something.another", {"df"}, 0],
        ["x = df.something()", {"df"}, 0],
        ['x = df["column"]', {"df"}, 0],
        ["x = df[another]", {"df", "another"}, 0],
        ["x = df[function(another)]", {"df", "function", "another"}, 0],
        ["df = load()\nx = df + another", {"load"}, 0],
        ["x = y + z", {"y", "z"}, 0],
        ["x = a + b + c + d", {"a", "b", "c", "d"}, 0],
        ["x = [y for y in range(10)]", set(), 0],
        ["x = np.std([y for y in range(10)])", {"np"}, 0],
    ],
    ids=[
        "attribute",
        "attribute-nested",
        "method",
        "getitem-literal",
        "getitem-variable",
        "getitem-nested",
        "multiline",
        "expression",
        "expression-long",
        "list-comprehension",
        "list-comprehension-as-arg",
    ],
)
def test_find_inputs_with_atom_expr(code, expected, index):
    atom_exp = get_first_sibling_after_assignment(code, index=index)
    assert io.find_inputs(atom_exp) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["[x for x in range(10)]", set()],
        ['[f"{x}" for x in range(10)]', set()],
        ["(x for x in range(10))", set()],
        ["[function(x) for x in range(10)]", {"function"}],
        ["[(x, y) for x, y in something(10)]", {"something"}],
        ["[x.attribute for x in range(10)]", set()],
        ["[x for x in obj if x > 0]", {"obj"}],
        ["[i for row in matrix for i in row]", {"matrix"}],
        ["[i for matrix in tensor for row in matrix for i in row]", {"tensor"}],
    ],
    ids=[
        "left-expression",
        "f-string",
        "generator",
        "both-expressions",
        "many-variables",
        "attributes",
        "conditional",
        "nested",
        "nested-double",
    ],
)
def test_find_list_comprehension_inputs(code, expected):
    tree = parso.parse(code)
    list_comp = tree.children[0].children[1]
    assert io.find_comprehension_inputs(list_comp) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["for i in range(10):\n    y = x + i", {"i"}],
        ["for i, j in something():\n    y = x + i", {"i", "j"}],
        ["def function(i):\n    y = x + i", {"i"}],
        ["def function(i, j):\n    y = x + i + j", {"i", "j"}],
    ],
    ids=[
        "for",
        "for-many",
        "def",
        "def-many",
    ],
)
def test_get_local_scope(code, expected):
    node = testutils.get_first_leaf_with_value(code, "x")
    assert io.get_local_scope(node) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ["a = 1", set()],
        ["a, b = 1, 2", set()],
        ["i = 1", {"i"}],
        ["a, i = 1, 2", {"i"}],
        ["i, b = 1, 2", {"i"}],
        ["(a, i) = 1, 2", {"i"}],
        ["(i, b) = 1, 2", {"i"}],
        ["[a, i] = 1, 2", {"i"}],
        ["[i, b] = 1, 2", {"i"}],
        ['[i["key"], b] = 1, 2', {"i"}],
        ["[i.attribute, b] = 1, 2", {"i"}],
        ["[i[key], b] = 1, 2", {"i"}],
        ["(i, (j, a)) = 1, (2, 3)", {"i", "j"}],
        ["(i, (j, (k, a))) = 1, (2, (3, 4))", {"i", "j", "k"}],
    ],
)
def test_get_modified_objects(code, expected):
    leaf = testutils.get_first_leaf_with_value(code, "=")
    assert io._get_modified_objects(leaf, {"i", "j", "k"}, set()) == expected
