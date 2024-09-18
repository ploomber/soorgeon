import pytest
from test_export import _read
from soorgeon import export
from ploomber.spec import DAGSpec

diff_cell_nb1 = """# ## Load df

df = 1

# ## load df again

df = 5

# ## load df-2

df_2 = df + 1

"""

diff_cell_nb2 = """
from sklearn.datasets import load_iris, load_digits

# ## Load df

df = load_digits(as_frame=True)['data']

# ## load df again

df = load_iris(as_frame=True)['data']

# ## load df-2

df_2 = df + 1

"""

same_cell_nb1 = """
from sklearn.datasets import load_iris, load_digits

# ## Loading df

df = load_digits(as_frame=True)['data']

# ## reload df and load df_2

df = load_iris(as_frame=True)['data']
df_2 = df + 1

"""


@pytest.mark.parametrize("nb", [same_cell_nb1])
def test_overriding_same_cell(nb):
    export.from_nb(_read(nb))

    dag = DAGSpec("pipeline.yaml").to_dag().render()
    assert set(dag["loading-df"].upstream) == set()
    assert set(dag["reload-df-and-load-df-2"].upstream) == set()


@pytest.mark.parametrize("nb", [diff_cell_nb1, diff_cell_nb2])
def test_overriding_diff_cell(nb):
    export.from_nb(_read(nb))

    dag = DAGSpec("pipeline.yaml").to_dag().render()

    assert set(dag["load-df"].upstream) == set()
    assert set(dag["load-df-again"].upstream) == set()
    assert set(dag["load-df-2"].upstream) == {"load-df-again"}
