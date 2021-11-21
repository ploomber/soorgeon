from pathlib import Path
import yaml
import parso
import pytest
import jupytext
from ploomber.spec import DAGSpec

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


@pytest.mark.parametrize('nb_str, tasks', [
    [simple, ['cell-0', 'cell-2', 'cell-4']],
    [simple_branch, ['first', 'second', 'third-a', 'third-b']],
    [eda, ['load', 'clean', 'plot']],
],
                         ids=[
                             'simple',
                             'simple-branch',
                             'eda',
                         ])
def test_from_nb(tmp_empty, nb_str, tasks):
    export.from_nb(_read(nb_str))

    dag = DAGSpec('pipeline.yaml').to_dag()

    dag.build()
    assert list(dag) == tasks


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

with_definitions_expected = ('## load\ndef load(x):\n    return x'
                             '\n\n## plot\ndef plot(x):\n    return x\n\n'
                             '## clean\nclass Cleaner:\n    pass')

definition_with_import = """# ## load

import matplotlib.pyplot as plt

def plot(x):
    plt.plot()


df = load()
"""

definition_with_import_expected = ('## load\nimport matplotlib.pyplot as plt'
                                   '\n\n\ndef plot(x):\n    plt.plot()')


@pytest.mark.parametrize('code, expected', [
    [with_definitions, with_definitions_expected],
    [definition_with_import, definition_with_import_expected],
])
def test_export_definitions(tmp_empty, code, expected):
    exporter = export.NotebookExporter(_read(code))
    exporter.export_definitions()

    assert Path('exported.py').read_text() == expected


def test_get_sources_includes_import_from_exported_definitions(tmp_empty):
    exporter = export.NotebookExporter(_read(with_definitions))

    sources = exporter.get_sources()

    import_ = 'from exported import load, plot, Cleaner'
    assert import_ in sources['load']
    assert import_ in sources['clean']
    assert import_ in sources['plot']
