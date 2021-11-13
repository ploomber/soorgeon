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

# ignore classes, functions
# try assigning a tuple


@pytest.mark.parametrize('code_str, inputs, outputs', [
    [only_outputs, set(), {'x', 'y'}],
    [simple, {'x', 'y'}, {'z'}],
    [local_inputs, set(), {'x', 'y', 'z'}],
    [imports, set(), {'z'}],
],
                         ids=[
                             'only_outputs',
                             'simple',
                             'local_inputs',
                             'imports',
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
