from pathlib import Path
import pytest
import json
from test_export import _read
from soorgeon import export
from ploomber.spec import DAGSpec

overridingVarNb1 = """# ## Load df

df = 1

# ## load df again

df = 5

# ## load df-2

df_2 = df + 1
assert df_2 == 6

"""

overridingVarNb2 = """
from sklearn.datasets import load_iris, load_digits

# ## Load df

df = load_digits(as_frame=True)['data']

# ## load df again

df = load_iris(as_frame=True)['data']

# ## load df-2

df_2 = df + 1
compare_df = load_iris(as_frame=True)['data']
compare_df = compare_df +1
assert df_2.equals(compare_df) == True

"""

# 1. generate a notebook file
# 2. Call soorgeon refactor
# 3. Run all the scripts in the task folder
# 4. df_2 should be 6 not 2; assert df_2 == 6
@pytest.mark.parametrize("nb", [overridingVarNb1, overridingVarNb2])
def test_overriding_vars(nb):
    # from_nb refactors a jupyter notebook
    export.from_nb(_read(nb))

    # run all three scrips in the tasks folder
    path = Path('tasks', 'load-df.ipynb')
    nb = path.read_text()
    nb = json.loads(nb)

    source1 = ""

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if not line.startswith('#'):
                    source1 += ''.join(line)
                    # hardcoded the product cells
                    if (line.find("product = None") != -1):
                        source1 += """product = {
                                \"df\": \"output/load-df-df.pkl\",
                                \"nb\": \"output/load-df.ipynb\",
                            }"""
            source1 += "\n"
    exec(source1)

    path = Path('tasks', 'load-df-again.ipynb')
    nb = path.read_text()
    nb = json.loads(nb)

    source2 = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if not line.startswith('#'):
                    source2 += ''.join(line)
                    if (line.find("product = None") != -1):
                        source2 += """product = {
                                \"df\": \"output/load-df-again-df.pkl\",
                                \"nb\": \"output/load-df-again.ipynb\",
                            }"""
            source2 += "\n"
    exec(source2)

    path = Path('tasks', 'load-df-2.ipynb')
    nb = path.read_text()
    nb = json.loads(nb)

    source3 = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if not line.startswith('#'):
                    source3 += ''.join(line)
                    if (line.find("product = None") != -1):
                        source3 += """\nupstream = {
                                \"load-df-again\": {
                                    \"df\": \"output/load-df-again-df.pkl\",
                                    \"nb\": \"output/load-df-again.ipynb\",
                                }
                            }
                            \nproduct = {
                                \"nb\": \"output/load-df-2.ipynb\"
                            }"""
            source3 += "\n"
    exec(source3)

dependency_nb = """ 
from sklearn.datasets import load_iris, load_digits

# ## Loading df

df = load_digits(as_frame=True)['data']

# ## reload df and load df_2

df = load_iris(as_frame=True)['data']
df_2 = df + 1

"""
@pytest.mark.parametrize("nb", [dependency_nb])
def test_dependency(nb):
    export.from_nb(_read(nb))

    spec = DAGSpec('pipeline.yaml').to_dag()
    expected = "upstream = None\nproduct = None"
    assert(spec["reload-df-and-load-df-2"].source._get_parameters_cell()) == expected