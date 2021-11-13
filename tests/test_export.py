import pytest
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
])
def test_from_nb(tmp_empty, nb_str, tasks):
    export.from_nb(_read(nb_str))

    dag = DAGSpec('pipeline.yaml').to_dag()

    dag.build()
    assert list(dag) == tasks
