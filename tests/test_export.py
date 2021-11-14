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


# FIXME: test does not serialize objects that arent used by downstream
# tasks

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
