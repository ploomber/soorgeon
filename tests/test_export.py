from pathlib import Path
from importlib import resources

import yaml
import parso
import pytest
import jupytext
from ploomber.spec import DAGSpec
import papermill as pm

from soorgeon import assets, export, exceptions, io


def _read(nb_str):
    return jupytext.reads(nb_str, fmt="py:light")


def _find_cells_with_tags(nb, tags):
    """
    Find the first cell with any of the given tags, returns a dictionary
    with 'cell' (cell object) and 'index', the cell index.
    """
    tags_to_find = list(tags)
    tags_found = {}

    for index, cell in enumerate(nb["cells"]):
        for tag in cell["metadata"].get("tags", []):
            if tag in tags_to_find:
                tags_found[tag] = dict(cell=cell, index=index)
                tags_to_find.remove(tag)

                if not tags_to_find:
                    break

    return tags_found


simple = """# ## Cell 0

x = 1

# ## Cell 2

y = x + 1

# ## Cell 4

z = y + 1
"""

simple_branch = """# ## First

x = 1

# ## Second

y = x + 1

# ## Third A

z = y + 1

# ## Third B

z2 = y + 1
"""

eda = """# # Some analysis
#
# ## Load

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris(as_frame=True)['data']

# ## Clean

df = df[df['petal length (cm)'] > 2]

# ## Plot

sns.histplot(df['petal length (cm)'])
"""

unused_products = """# ## Cell 0

x = 1
x2 = 2

# ## Cell 2

y = x + 1

# ## Cell 4

z = y + 1
"""

complex = """\
# ## one

with open('one-1', 'w') as f:
    with open('one-2', 'w') as g:
        f.write(''), g.write('')

x, xx = 1, 2

matrix = [range(10) for _ in range(10)]
numbers = [i for row in matrix for i in row]

# ## two


with open('one-1', 'w') as f, open('one-2', 'w') as g:
    f.write(''), g.write('')

y = 1 + 1

print(f'{x} {y!r} {x:.2f}')

for n in range(xx):
    for m in range(y):
        with open('file', 'w') as f:
            f.write(f'{n} {m}')


matrix_another = [range(10) for _ in range(10)]
[i + x + y for row in matrix for i in row]

# ## three

try:
    raise ValueError
except Exception as e:
    print('something happened')

    if False:
        raise Exception from e

z = y + 1

stuff = [f"'{s}'" for s in [] if s not in []]

a_, b_ = range(10), range(10)
things = {f'"{a}"': b for a, b in zip(a_, b_) if b > 3}
"""

# NOTE: since we turn magics into comments, we add an import in the middle
# of the notebook to ensure that the magic above it won't turn into a comment
# in the final import node (the import note will be moved to the top of
# the notebook)
magics = """\
# ## first

# + language="bash"
# ls

# + language="html"
# <br>hi
# -

import math
math.sqrt(1)

# ## second

# %timeit 1 + 1

# %cd x

# %%capture
print('x')
"""

magics_structured = """\
# ## first

def do():
    pass

# + language="bash"
# ls

# + language="html"
# <br>hi
# -

# ## second

# %timeit do()

# %timeit do()

# +
x = 1
y = 2
# -

# ## third

# %%capture
print('something')

z = x + y
"""


@pytest.mark.parametrize(
    "nb_str, tasks",
    [
        [simple, ["cell-0", "cell-2", "cell-4"]],
        [simple_branch, ["first", "second", "third-a", "third-b"]],
        [eda, ["load", "clean", "plot"]],
        [complex, ["one", "two", "three"]],
        [magics, ["first", "second"]],
        [magics_structured, ["first", "second", "third"]],
    ],
    ids=[
        "simple",
        "simple-branch",
        "eda",
        "complex",
        "magics",
        "magics-structured",
    ],
)
def test_from_nb(tmp_empty, nb_str, tasks):
    export.from_nb(_read(nb_str), py=True)

    dag = DAGSpec("pipeline.yaml").to_dag()

    dag.build()
    assert list(dag) == tasks


@pytest.mark.parametrize(
    "py, ext",
    [
        [True, "py"],
        [False, "ipynb"],
    ],
    ids=[
        "py",
        "ipynb",
    ],
)
def test_from_nb_works_with_magics(tmp_empty, py, ext):
    export.from_nb(_read(magics), py=py)

    first = jupytext.read(Path("tasks", f"first.{ext}"))
    second = jupytext.read(Path("tasks", f"second.{ext}"))

    assert [c["source"] for c in first.cells] == [
        "import math",
        "upstream = None\nproduct = None",
        "## first",
        "%%bash\nls",
        "%%html\n<br>hi",
        "\nmath.sqrt(1)",
    ]

    assert [c["source"] for c in second.cells] == [
        "upstream = None\nproduct = None",
        "## second",
        "%timeit 1 + 1",
        "%cd x",
        "%%capture\nprint('x')",
    ]


def test_exporter_infers_structure_from_line_magics():
    exporter = export.NotebookExporter(_read(magics_structured))

    assert set(exporter.get_sources()) == {"first", "second", "third"}
    assert io.find_upstream(exporter._snippets) == {
        "first": [],
        "second": [],
        "third": ["second"],
    }
    assert exporter.io == {
        "first": (set(), set()),
        "second": (set(), {"x", "y"}),
        "third": ({"x", "y"}, set()),
    }


def test_from_nb_with_star_imports(tmp_empty):
    nb_str = """\
# ## Some header

from math import *

# ## Another header

from pathlib import *
"""

    with pytest.raises(exceptions.InputError) as excinfo:
        export.from_nb(_read(nb_str), py=True)

    assert "from math import *" in str(excinfo.value)
    assert "from pathlib import *" in str(excinfo.value)


def test_from_nb_upstream_cell_only_shows_unique_values(tmp_empty):
    export.from_nb(_read(complex))

    dag = DAGSpec("pipeline.yaml").to_dag()

    expected = "upstream = ['one']\nproduct = None"
    assert dag["two"].source._get_parameters_cell() == expected


def test_from_nb_with_product_prefix(tmp_empty):
    export.from_nb(_read(simple), product_prefix="some-directory")

    dag = DAGSpec("pipeline.yaml").to_dag()

    products = [
        i
        for meta in (t.product.to_json_serializable().values() for t in dag.values())
        for i in meta
    ]

    expected = str(Path(tmp_empty, "some-directory"))
    assert all([p.startswith(expected) for p in products])


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ["some-directory", "some-directory\n"],
        [None, "output\n"],
    ],
)
def test_from_nb_creates_gitignore(tmp_empty, prefix, expected):
    export.from_nb(_read(simple), product_prefix=prefix)

    assert Path(".gitignore").read_text() == expected


def test_from_nb_appends_gitignore(tmp_empty):
    path = Path(".gitignore")
    path.write_text("something")

    export.from_nb(_read(simple), product_prefix="some-directory")

    assert path.read_text() == "something\nsome-directory\n"


def test_from_nb_doesnt_create_gitignore_if_absolute_prefix(tmp_empty):
    export.from_nb(_read(simple), product_prefix="/some/absolute/dir")

    assert not Path(".gitignore").exists()


def test_from_nb_doesnt_append_gitignore_if_absolute_prefix(tmp_empty):
    path = Path(".gitignore")
    path.write_text("something")

    export.from_nb(_read(simple), product_prefix="/some/absolute/dir")

    assert path.read_text() == "something"


def test_spec_style(tmp_empty):
    export.from_nb(_read(simple))
    spec = Path("pipeline.yaml").read_text()
    d = yaml.safe_load(spec)

    # check empty space between tasks
    assert "\n\n-" in spec
    # check source is the first key on every task
    assert all([list(spec)[0] == "source" for spec in d["tasks"]])


def test_from_nb_does_not_serialize_unused_products(tmp_empty):
    export.from_nb(_read(unused_products))

    dag = DAGSpec("pipeline.yaml").to_dag()

    assert set(k for k in dag["cell-0"].product.to_json_serializable()) == {
        "nb",
        "x",
    }


# TODO: test all expected tags appear


@pytest.fixture
def eda_sources():
    exporter = export.NotebookExporter(_read(eda), py=True)
    sources = exporter.get_sources()
    return sources


def test_exporter_removes_imports(eda_sources):
    nb = jupytext.reads(eda_sources["load"], fmt="py:percent")

    # imports should only exist in the soorgeon-imports cell
    m = _find_cells_with_tags(nb, ["soorgeon-imports"])
    nb.cells.pop(m["soorgeon-imports"]["index"])
    tree = parso.parse(jupytext.writes(nb, fmt="py:percent"))

    assert not list(tree.iter_imports())


def test_exporter_does_not_add_unpickling_if_no_upstream(eda_sources):
    nb = jupytext.reads(eda_sources["load"], fmt="py:percent")
    assert not _find_cells_with_tags(nb, ["soorgeon-unpickle"])


# FIXME: another test but when we have outputs but they're not used
def test_exporter_does_not_add_pickling_if_no_outputs(eda_sources):
    nb = jupytext.reads(eda_sources["plot"], fmt="py:percent")
    assert not _find_cells_with_tags(nb, ["soorgeon-pickle"])


with_definitions = """# ## load

def load(x):
    return x

1 + 1

# ## clean

class Cleaner:
    pass


2 + 2

# ## plot

def plot(x):
    return x

df = load(1)
"""

with_definitions_expected = (
    "def load(x):\n    return x\n\ndef plot(x):\n    return x\n\n"
    "class Cleaner:\n    pass"
)

definition_with_import = """
# ## load

import matplotlib.pyplot as plt
import load

def plot(x):
    plt.plot()


df = load()
"""

definition_with_import_expected = (
    "import matplotlib.pyplot as plt" "\n\n\ndef plot(x):\n    plt.plot()"
)


@pytest.mark.parametrize(
    "code, expected",
    [
        [with_definitions, with_definitions_expected],
        [definition_with_import, definition_with_import_expected],
    ],
    ids=[
        "with_definitions",
        "definition_with_import",
    ],
)
def test_export_definitions(tmp_empty, code, expected):
    exporter = export.NotebookExporter(_read(code))
    exporter.export_definitions()

    assert Path("exported.py").read_text() == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        [with_definitions, "ploomber>=0.14.7\n"],
        [definition_with_import, "load\nmatplotlib\nploomber>=0.14.7\n"],
    ],
    ids=[
        "with_definitions",
        "definition_with_import",
    ],
)
def test_export_requirements(tmp_empty, code, expected):
    exporter = export.NotebookExporter(_read(code))
    exporter.export_requirements()

    expected = "# Auto-generated file, may need manual " f"editing\n{expected}"
    assert Path("requirements.txt").read_text() == expected


def test_export_requirements_doesnt_overwrite(tmp_empty):
    reqs = Path("requirements.txt")
    reqs.write_text("soorgeon\n")

    exporter = export.NotebookExporter(_read(definition_with_import))
    exporter.export_requirements()

    expected = (
        "soorgeon\n# Auto-generated file, may need manual "
        "editing\nload\nmatplotlib\nploomber>=0.14.7\n"
    )
    assert reqs.read_text() == expected


def test_does_not_create_exported_py_if_no_definitions(tmp_empty):
    exporter = export.NotebookExporter(_read(simple))
    exporter.export_definitions()

    assert not Path("exported.py").exists()


def test_get_sources_includes_import_from_exported_definitions(tmp_empty):
    exporter = export.NotebookExporter(_read(with_definitions))

    sources = exporter.get_sources()

    import_ = "from exported import load, plot, Cleaner"
    assert import_ in sources["load"]
    assert import_ in sources["clean"]
    assert import_ in sources["plot"]


for_loop_with_output_in_body = """# ## section

def fn():
    print(x)


x = 1

fn()
"""


def test_raise_an_error_if_function_uses_global_variables():
    nb = _read(for_loop_with_output_in_body)

    with pytest.raises(exceptions.InputError) as excinfo:
        export.NotebookExporter(nb)

    assert "Function 'fn' uses variables 'x'" in str(excinfo.value)


# FIXME: test logging option

list_comp = """
# ## first

x = [1, 2, 3]

[y for y in x]
"""


@pytest.mark.parametrize(
    "code, expected",
    [
        [list_comp, {"first": (set(), {"x"})}],
    ],
)
def test_get_raw_io(code, expected):
    nb = jupytext.reads(code, fmt="py:light")
    exporter = export.NotebookExporter(nb)

    assert exporter._get_raw_io() == expected


def test_exporter_init_with_syntax_error():
    code = """\
# ## first

if
"""
    nb = jupytext.reads(code, fmt="py:light")

    with pytest.raises(exceptions.InputSyntaxError):
        export.NotebookExporter(nb)


def test_exporter_init_with_undefined_name_error():
    code = """\
# ## first

y = x + 1
"""
    nb = jupytext.reads(code, fmt="py:light")

    with pytest.raises(exceptions.InputWontRunError) as excinfo:
        export.NotebookExporter(nb)

    expected = (
        "(ensure that your notebook executes from " "top-to-bottom and try again)"
    )
    assert expected in str(excinfo.value)


def test_get_code(tmp_empty):
    code = """\
# ## first

# ### this should not appear since its a markdown cell

print('hello')
"""
    nb_ = jupytext.reads(code, fmt="py:light")
    jupytext.write(nb_, "nb.ipynb")
    pm.execute_notebook("nb.ipynb", "nb.ipynb", kernel_name="python3")

    nb = jupytext.read("nb.ipynb")
    exporter = export.NotebookExporter(nb)

    assert exporter._get_code() == "print('hello')"


def test_get_sources_add_import_if_needed():
    code = """\
# ## first

import something

x = something.do()

# ## second

y = something.another()
"""
    nb = jupytext.reads(code, fmt="py:light")
    exporter = export.NotebookExporter(nb)
    sources = exporter.get_sources()

    assert "import something" in sources["first"]
    assert "import something" in sources["second"]


def test_get_task_specs():
    code = """\
# ## first

import something

x = something.do()

# ## second

y = x + something.another()
"""
    nb = jupytext.reads(code, fmt="py:light")
    exporter = export.NotebookExporter(nb, py=True)
    specs = exporter.get_task_specs(product_prefix="output")

    assert specs == {
        "first": {
            "source": "tasks/first.py",
            "product": {"x": "output/first-x.pkl", "nb": "output/first.ipynb"},
        },
        "second": {
            "source": "tasks/second.py",
            "product": {"nb": "output/second.ipynb"},
        },
    }


def test_check_functions_do_not_use_global_variables():
    code = """
def my_function(a, b, c=None):
    return a + b + c
"""

    export._check_functions_do_not_use_global_variables(code)


# FIXME: this is broken because we consider all the definitions in the file
# but we should only take into account the ones that happen before the node
# we're parsing
def test_check_functions_do_not_use_global_variables_exception():
    code = """
def my_function(a, b):
    return a + b + c

class c:
    pass
"""

    export._check_functions_do_not_use_global_variables(code)


none_pickling = """\
Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
Path(product['df']).write_bytes(pickle.dumps(df))

Path(product['x']).parent.mkdir(exist_ok=True, parents=True)
Path(product['x']).write_bytes(pickle.dumps(x))\
"""

none_unpickling = """\
df = pickle.loads(Path(upstream['first']['df']).read_bytes())
x = pickle.loads(Path(upstream['first']['x']).read_bytes())\
"""

parquet_pickling = """\
Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
df.to_parquet(product['df'], index=False)

Path(product['x']).parent.mkdir(exist_ok=True, parents=True)
Path(product['x']).write_bytes(pickle.dumps(x))\
"""

parquet_unpickling = """\
df = pd.read_parquet(upstream['first']['df'])
x = pickle.loads(Path(upstream['first']['x']).read_bytes())\
"""

csv_pickling = """\
Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
df.to_csv(product['df'], index=False)

Path(product['x']).parent.mkdir(exist_ok=True, parents=True)
Path(product['x']).write_bytes(pickle.dumps(x))\
"""

csv_unpickling = """\
df = pd.read_csv(upstream['first']['df'])
x = pickle.loads(Path(upstream['first']['x']).read_bytes())\
"""


@pytest.mark.parametrize(
    "df_format, pickling, unpickling",
    [
        [None, none_pickling, none_unpickling],
        ["parquet", parquet_pickling, parquet_unpickling],
        ["csv", csv_pickling, csv_unpickling],
    ],
    ids=[
        "none",
        "parquet",
        "csv",
    ],
)
def test_prototask_un_pickling_cells(df_format, pickling, unpickling):
    code = """\
# ## first

import something

df = 1
x = 1

# ## second

df_2 = x + df + 1
"""
    exporter = export.NotebookExporter(_read(code), df_format=df_format)
    one, two = exporter._proto_tasks

    assert one._pickling_cell(exporter.io)["source"] == pickling
    assert two._pickling_cell(exporter.io) is None

    assert one._unpickling_cell(exporter.io, exporter.providers) is None
    assert two._unpickling_cell(exporter.io, exporter.providers)["source"] == unpickling


cloudpickle_pickling = """\
Path(product['x']).parent.mkdir(exist_ok=True, parents=True)
Path(product['x']).write_bytes(cloudpickle.dumps(x))\
"""

cloudpickle_unpickling = """\
x = cloudpickle.loads(Path(upstream['first']['x']).read_bytes())\
"""

dill_pickling = """\
Path(product['x']).parent.mkdir(exist_ok=True, parents=True)
Path(product['x']).write_bytes(dill.dumps(x))\
"""

dill_unpickling = """\
x = dill.loads(Path(upstream['first']['x']).read_bytes())\
"""


@pytest.mark.parametrize(
    "serializer, pickling, unpickling",
    [
        ["cloudpickle", cloudpickle_pickling, cloudpickle_unpickling],
        ["dill", dill_pickling, dill_unpickling],
    ],
    ids=["cloudpickle", "dill"],
)
def test_prototask_un_pickling_cells_with_serializer(serializer, pickling, unpickling):
    code = """\
# ## first

import something

x = lambda n : n ** 2

# ## second

x = x(10)
x = x + 2
"""
    exporter = export.NotebookExporter(_read(code), serializer=serializer)
    one, two = exporter._proto_tasks

    assert one._pickling_cell(exporter.io)["source"] == pickling
    assert two._pickling_cell(exporter.io)["source"] == pickling

    assert one._unpickling_cell(exporter.io, exporter.providers) is None
    assert two._unpickling_cell(exporter.io, exporter.providers)["source"] == unpickling


def test_validates_df_format():
    with pytest.raises(ValueError) as excinfo:
        export.NotebookExporter(_read(""), df_format="something")

    assert "df_format must be one of " in str(excinfo.value)


def test_validates_serializer():
    with pytest.raises(ValueError) as excinfo:
        export.NotebookExporter(_read(""), serializer="something")

    assert "serializer must be one of " in str(excinfo.value)


def test_creates_readme(tmp_empty):
    exporter = export.NotebookExporter(_read(simple))
    exporter.export()

    assert Path("README.md").read_text() == resources.read_text(assets, "README.md")


def test_appends_to_readme(tmp_empty):
    Path("README.md").write_text("# Some stuff")
    exporter = export.NotebookExporter(_read(simple))
    exporter.export()

    expected = "# Some stuff\n" + resources.read_text(assets, "README.md")
    assert Path("README.md").read_text() == expected


@pytest.mark.parametrize(
    "code, expect",
    [
        ("f = open('text.txt')", False),
        ("f = open('read.txt' , 'r')", False),
        ("f = open('txt', 'r')", False),
        ("f = open('text.txt',  'rb') ", False),
        ("f = open('text.txt'  ,   'ab')", True),
        ("with open('text.txt',   'w')", True),
        ("with open('txt' ,  'w+')", True),
        ("''' with open('txt' ,  'w+') '''", False),
        ("f = Path().write_text()", True),
        ("f = path().write_bytes()", True),
        ("df.to_csv()", True),
        ("df.to_parquet()", True),
        ("write_text = 6", False),
        ("header = 'call to_csv function'", False),
        ("# Path.write_text('txt')", False),
    ],
)
def test_find_output_file_events(code, expect):
    actual = export._find_output_file_events(code)
    assert actual == expect
