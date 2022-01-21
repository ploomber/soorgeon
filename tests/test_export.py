from pathlib import Path
import yaml
import parso
import pytest
import jupytext
from ploomber.spec import DAGSpec
import papermill as pm

from soorgeon import export


def _read(nb_str):
    return jupytext.reads(nb_str, fmt='py:light')


def _find_cells_with_tags(nb, tags):
    """
    Find the first cell with any of the given tags, returns a dictionary
    with 'cell' (cell object) and 'index', the cell index.
    """
    tags_to_find = list(tags)
    tags_found = {}

    for index, cell in enumerate(nb['cells']):
        for tag in cell['metadata'].get('tags', []):
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


@pytest.mark.parametrize('nb_str, tasks', [
    [simple, ['cell-0', 'cell-2', 'cell-4']],
    [simple_branch, ['first', 'second', 'third-a', 'third-b']],
    [eda, ['load', 'clean', 'plot']],
    [complex, ['one', 'two', 'three']],
],
                         ids=[
                             'simple',
                             'simple-branch',
                             'eda',
                             'complex',
                         ])
def test_from_nb(tmp_empty, nb_str, tasks):
    export.from_nb(_read(nb_str))

    dag = DAGSpec('pipeline.yaml').to_dag()

    dag.build()
    assert list(dag) == tasks


def test_from_nb_with_product_prefix(tmp_empty):
    export.from_nb(_read(simple), product_prefix='some-directory')

    dag = DAGSpec('pipeline.yaml').to_dag()

    products = [
        i for meta in (t.product.to_json_serializable().values()
                       for t in dag.values()) for i in meta
    ]

    expected = str(Path(tmp_empty, 'some-directory'))
    assert all([p.startswith(expected) for p in products])


def test_spec_style(tmp_empty):
    export.from_nb(_read(simple))
    spec = Path('pipeline.yaml').read_text()
    d = yaml.safe_load(spec)

    # check empty space between tasks
    assert '\n\n-' in spec
    # check source is the first key on every task
    assert all([list(spec)[0] == 'source' for spec in d['tasks']])


def test_from_nb_does_not_serialize_unused_products(tmp_empty):
    export.from_nb(_read(unused_products))

    dag = DAGSpec('pipeline.yaml').to_dag()

    assert set(k for k in dag['cell-0'].product.to_json_serializable()) == {
        'nb',
        'x',
    }


# TODO: test all expected tags appear


@pytest.fixture
def eda_sources():
    exporter = export.NotebookExporter(_read(eda))
    sources = exporter.get_sources()
    return sources


def test_exporter_removes_imports(eda_sources):
    nb = jupytext.reads(eda_sources['load'], fmt='py:percent')

    # imports should only exist in the soorgeon-imports cell
    m = _find_cells_with_tags(nb, ['soorgeon-imports'])
    nb.cells.pop(m['soorgeon-imports']['index'])
    tree = parso.parse(jupytext.writes(nb, fmt='py:percent'))

    assert not list(tree.iter_imports())


def test_exporter_does_not_add_unpickling_if_no_upstream(eda_sources):
    nb = jupytext.reads(eda_sources['load'], fmt='py:percent')
    assert not _find_cells_with_tags(nb, ['soorgeon-unpickle'])


# FIXME: another test but when we have outputs but they're not used
def test_exporter_does_not_add_pickling_if_no_outputs(eda_sources):
    nb = jupytext.reads(eda_sources['plot'], fmt='py:percent')
    assert not _find_cells_with_tags(nb, ['soorgeon-pickle'])


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
    'def load(x):\n    return x\n\ndef plot(x):\n    return x\n\n'
    'class Cleaner:\n    pass')

definition_with_import = """# ## load

import matplotlib.pyplot as plt

def plot(x):
    plt.plot()


df = load()
"""

definition_with_import_expected = ('import matplotlib.pyplot as plt'
                                   '\n\n\ndef plot(x):\n    plt.plot()')


@pytest.mark.parametrize('code, expected', [
    [with_definitions, with_definitions_expected],
    [definition_with_import, definition_with_import_expected],
],
                         ids=['with_definitions', 'definition_with_import'])
def test_export_definitions(tmp_empty, code, expected):
    exporter = export.NotebookExporter(_read(code))
    exporter.export_definitions()

    assert Path('exported.py').read_text() == expected


@pytest.mark.parametrize('code, expected', [
    [with_definitions, None],
    [definition_with_import, 'matplotlib\n'],
],
                         ids=[
                             'with_definitions',
                             'definition_with_import',
                         ])
def test_export_requirements(tmp_empty, code, expected):
    exporter = export.NotebookExporter(_read(code))
    exporter.export_requirements()

    if expected is None:
        assert not Path('requirements.txt').exists()
    else:
        expected = ('# Auto-generated file, may need manual '
                    f'editing\n{expected}')
        assert Path('requirements.txt').read_text() == expected


def test_does_not_create_exported_py_if_no_definitions(tmp_empty):
    exporter = export.NotebookExporter(_read(simple))
    exporter.export_definitions()

    assert not Path('exported.py').exists()


def test_get_sources_includes_import_from_exported_definitions(tmp_empty):
    exporter = export.NotebookExporter(_read(with_definitions))

    sources = exporter.get_sources()

    import_ = 'from exported import load, plot, Cleaner'
    assert import_ in sources['load']
    assert import_ in sources['clean']
    assert import_ in sources['plot']


for_loop_with_output_in_body = """# ## section

def fn():
    print(x)


x = 1

fn()
"""


def test_raise_an_error_if_function_uses_global_variables():
    nb = _read(for_loop_with_output_in_body)

    with pytest.raises(ValueError) as excinfo:
        export.NotebookExporter(nb)

    assert "Function 'fn' uses variables 'x'" in str(excinfo.value)


# FIXME: test logging option

list_comp = """
# ## first

x = [1, 2, 3]

[y for y in x]
"""


@pytest.mark.parametrize('code, expected', [
    [list_comp, {
        'first': (set(), {'x'})
    }],
])
def test_get_raw_io(code, expected):
    nb = jupytext.reads(code, fmt='py:light')
    exporter = export.NotebookExporter(nb)

    assert exporter._get_raw_io() == expected


def test_exporter_init_with_syntax_error():
    code = """\
# ## first

if
"""
    nb = jupytext.reads(code, fmt='py:light')

    with pytest.raises(SyntaxError):
        export.NotebookExporter(nb)


def test_get_code(tmp_empty):
    code = """\
# ## first

# ### this should not appear since its a markdown cell

print('hello')
"""
    nb_ = jupytext.reads(code, fmt='py:light')
    jupytext.write(nb_, 'nb.ipynb')
    pm.execute_notebook('nb.ipynb', 'nb.ipynb', kernel_name='python3')

    nb = jupytext.read('nb.ipynb')
    exporter = export.NotebookExporter(nb)

    assert exporter._get_code() == "print('hello')"


def test_get_sources():
    code = """\
# ## first

import something

x = something.do()

# ## second

y = something.another()
"""
    nb = jupytext.reads(code, fmt='py:light')
    exporter = export.NotebookExporter(nb)
    sources = exporter.get_sources()

    assert 'import something' in sources['first']
    assert 'import something' in sources['second']


def test_get_task_specs():
    code = """\
# ## first

import something

x = something.do()

# ## second

y = x + something.another()
"""
    nb = jupytext.reads(code, fmt='py:light')
    exporter = export.NotebookExporter(nb)
    specs = exporter.get_task_specs()

    assert specs == {
        'first': {
            'source': 'tasks/first.py',
            'product': {
                'x': 'output/first-x.pkl',
                'nb': 'output/first.ipynb'
            }
        },
        'second': {
            'source': 'tasks/second.py',
            'product': {
                'nb': 'output/second.ipynb'
            }
        }
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
@pytest.mark.xfail
def test_check_functions_do_not_use_global_variables_exception():
    code = """
def my_function(a, b):
    return a + b + c

class c:
    pass
"""

    export._check_functions_do_not_use_global_variables(code)
