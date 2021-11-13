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

# ignore classes, functions
# try assigning a tuple


@pytest.mark.parametrize('code_str, inputs, outputs', [
    [only_outputs, set(), {'x', 'y'}],
    [simple, {'x', 'y'}, {'z'}],
    [local_inputs, set(), {'x', 'y', 'z'}],
    [imports, set(), {'z'}],
    [imported_function, set(), {'df'}],
    [input_in_function_call, {'df'}, set()],
],
                         ids=[
                             'only_outputs',
                             'simple',
                             'local_inputs',
                             'imports',
                             'imported_function',
                             'input_in_function_call',
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
