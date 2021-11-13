import jupytext

from ploomber.spec import DAGSpec
from soorgeon import export


def _read(nb_str):
    return jupytext.reads(nb_str, fmt='py:light')


simple = """# ## Cell 0

x = 1

# ## Cell 2

y = x + 1

# ## Cell 4

z = y + 1
"""


def test_from_nb(tmp_empty):
    export.from_nb(_read(simple))

    dag = DAGSpec('pipeline.yaml').to_dag()

    dag.build()
    assert list(dag) == ['cell-0', 'cell-2', 'cell-4']
