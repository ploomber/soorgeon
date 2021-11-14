from conftest import read_snippets

import parso
import pytest

from soorgeon import static_analysis

only_outputs = """
x = 1
y = 2
"""

simple = """
z = x + y
"""

local_inputs = """
x = 1
y = 2
z = x + y
"""

imports = """
import pandas as pd

z = 1
"""

imported_function = """
from sklearn.datasets import load_iris

# load_iris should be considered an input since it's an imported object
df = load_iris(as_frame=True)['data']
"""

# FIXME: another test case but with a class constructor
input_in_function_call = """
sns.histplot(df.some_column)
"""

input_existing_object = """
X = 1
sns.histplot(X)
"""

# ignore classes, functions
# try assigning a tuple

# TODO: same but assigning multiple e.g., a, b = dict(), dict()
built_in = """
mapping = dict()
mapping['key'] = 'value'
"""

# TODO: same but with dot access
modify_existing_obj_getitem = """
mapping = {'a': 1}
mapping['key'] = 'value'
"""

# TODO: same but with dot access
modify_imported_obj_getitem = """
from pkg import mapping

mapping['key'] = 'value'
"""

define_multiple_outputs = """
a, b, c = 1, 2, 3
"""


@pytest.mark.parametrize('code_str, inputs, outputs', [
    [only_outputs, set(), {'x', 'y'}],
    [simple, {'x', 'y'}, {'z'}],
    [local_inputs, set(), {'x', 'y', 'z'}],
    [imports, set(), {'z'}],
    [imported_function, set(), {'df'}],
    [input_in_function_call, {'df'}, set()],
    [modify_existing_obj_getitem,
     set(), {'mapping'}],
    [modify_imported_obj_getitem, set(),
     set()],
    [built_in, set(), {'mapping'}],
    [input_existing_object, set(), {'X'}],
    [define_multiple_outputs, set(), {'a', 'b', 'c'}],
],
                         ids=[
                             'only_outputs',
                             'simple',
                             'local_inputs',
                             'imports',
                             'imported_function',
                             'input_in_function_call',
                             'modify_existing_getitem',
                             'modify_imported_getitem',
                             'built_in',
                             'input_existing_object',
                             'define_multiple_outputs',
                         ])
def test_find_inputs_and_outputs(code_str, inputs, outputs):
    in_, out = static_analysis.find_inputs_and_outputs(code_str)

    assert in_ == inputs
    assert out == outputs


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
    'load': "import load_data, plot\ndf = load_data()",
    # note that we re-define df
    'clean': "df = df[df.some_columns > 2]",
    'plot': "plot(df)"
}


def test_providermapping():
    m = static_analysis.ProviderMapping(static_analysis.find_io(eda))

    assert m._providers_for_task('load') == {}
    assert m._providers_for_task('clean') == {'df': 'load'}
    assert m._providers_for_task('plot') == {'df': 'clean'}
    assert m.get('df', 'clean') == 'load'


@pytest.mark.parametrize('snippets, expected', [
    [{
        'first': first,
        'second': second,
        'third': third
    }, {
        'first': [],
        'second': ['first'],
        'third': ['second']
    }],
    [eda, {
        'load': [],
        'clean': ['load'],
        'plot': ['clean']
    }],
])
def test_find_upstream(snippets, expected):
    assert static_analysis.find_upstream(snippets) == expected


simple_imports = """
import pandas as pd
import numpy as np
"""


@pytest.mark.parametrize('code, expected', [
    [
        simple_imports, {
            'np': 'import numpy as np',
            'pd': '\nimport pandas as pd'
        }
    ],
])
def test_find_defined_names_from_imports(code, expected):
    assert static_analysis.find_defined_names_from_imports(
        parso.parse(code)) == expected


@pytest.mark.parametrize('snippets, expected', [
    [
        eda, {
            'load': (set(), {'df'}),
            'clean': ({'df'}, {'df'}),
            'plot': ({'df'}, set())
        }
    ],
    [
        read_snippets('ml'),
        {
            'load': (set(), {'df', 'ca_housing'}),
            'clean': ({'df'}, {'df'}),
            'train-test-split':
            ({'df'}, {'y', 'X', 'X_train', 'X_test', 'y_train', 'y_test'}),
            'linear-regression':
            ({'y_test','X_test', 'y_train', 'X_train'}, {'lr', 'y_pred'}),
            'random-forest-regressor':
            ({'y_test','X_test', 'y_train', 'X_train'}, {'rf', 'y_pred'})
        },
    ],
],
                         ids=[
                             'eda',
                             'ml',
                         ])
def test_find_io(snippets, expected):
    assert static_analysis.find_io(snippets) == expected


@pytest.mark.parametrize('code, expected', [
    ['sns.histplot(df.some_column)', True],
    ['histplot(df.some_column)', True],
    ['sns.histplot(df)', True],
    ['histplot(df)', True],
    ['sns.histplot(df["key"])', True],
])
def test_inside_function_call(code, expected):
    leaf = parso.parse(code).get_first_leaf()

    while leaf:
        if leaf.value == 'df':
            break

        leaf = leaf.get_next_leaf()

    assert static_analysis.inside_function_call(leaf) is expected


exploratory = """
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris(as_frame=True)['data']

df = df[df['petal length (cm)'] > 2]

sns.histplot(df['petal length (cm)'])
"""


@pytest.mark.parametrize('code_nb, code_task, expected', [
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
])
def test_importsparser(code_nb, code_task, expected):
    ip = static_analysis.ImportsParser(code_nb)
    assert ip.get_imports_cell_for_task(code_task) == expected


@pytest.mark.parametrize('code, expected', [
    ['import pandas as pd\nimport numpy as np', '\n'],
    ['import math', ''],
    ['import pandas as pd\n1+1', '\n1+1'],
    ['import math\n1+1', '\n1+1'],
])
def test_remove_imports(code, expected):
    assert static_analysis.remove_imports(code) == expected