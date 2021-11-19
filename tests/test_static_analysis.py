from conftest import read_snippets

import parso
import pytest

from soorgeon import static_analysis


def get_first_leaf_with_value(code, value):
    leaf = parso.parse(code).get_first_leaf()

    while leaf:
        if leaf.value == value:
            return leaf

        leaf = leaf.get_next_leaf()

    raise ValueError(f'could not find leaf with value {value}')


def get_first_sibling_after_assignment(code, index):
    tree = parso.parse(code)
    leaf = tree.get_first_leaf()
    current = 0

    while leaf:
        if leaf.value == '=':
            if index == current:
                break

            current += 1

        leaf = leaf.get_next_leaf()

    return leaf.get_next_sibling()


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
import seaborn as sns

sns.histplot(df.some_column)
"""

# TODO: try all combinations of the following examples
input_key_in_function_call = """
import seaborn as sns

sns.histplot(x=df)
"""

input_key_in_function_call_many = """
import seaborn as sns

sns.histplot(x=df, y=df_another)
"""

input_key_in_function_call_with_dot_access = """
import seaborn as sns

sns.histplot(x=df.some_column)
"""

input_existing_object = """
import seaborn as sns

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

built_in_as_arg = """
from pkg import some_fn

something = some_fn(int)
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

local_function = """
def x():
    pass

y = x()
"""

local_function_with_args = """
def x(z):
    pass

y = x(10)
"""

local_function_with_args_and_body = """
def x(z):
    another = z + 1
    return another

y = x(10)
"""

local_class = """
class X:
    pass

y = X()
"""

for_loop = """
for x in range(10):
    y = x + z
"""

for_loop_many = """
for x, z in range(10):
    y = x + z
"""

for_loop_nested = """
for a, (b, (c, d)) in range(10):
    x = a + b + c + d
"""

for_loop_name_reference = """
for _, source in enumerate(10):
    some_function('%s' % source)
"""

# TODO: try with other variables such as accessing an attribute,
# or even just having the variable there, like "df"
getitem_input = """
df['x'].plot()
"""

method_access_input = """
df.plot()
"""

overriding_name = """
from pkg import some_function

x, y = some_function(x, y)
"""

# FIXME: test case with global scoped variables accessed in function/class
# definitions
"""
def function(x):
    # df may come from another task!
    return df + x

"""

# TODO: define inputs inside built-ins
# e.g.
# models = [a, b, c]
# models = {'a': a}

# TODO: we need a general function that finds the names after an =
# e.g. a = something(x=1, b=something)
# a = dict(a=1)
# b = {'a': x}


@pytest.mark.parametrize('code_str, inputs, outputs', [
    [only_outputs, set(), {'x', 'y'}],
    [simple, {'x', 'y'}, {'z'}],
    [local_inputs, set(), {'x', 'y', 'z'}],
    [imports, set(), {'z'}],
    [imported_function, set(), {'df'}],
    [input_in_function_call, {'df'}, set()],
    [input_key_in_function_call, {'df'},
     set()],
    [input_key_in_function_call_many, {'df', 'df_another'},
     set()],
    [input_key_in_function_call_with_dot_access, {'df'},
     set()],
    [modify_existing_obj_getitem,
     set(), {'mapping'}],
    [modify_imported_obj_getitem, set(),
     set()],
    [built_in, set(), {'mapping'}],
    [built_in_as_arg, set(), {'something'}],
    [input_existing_object, set(), {'X'}],
    [define_multiple_outputs, set(), {'a', 'b', 'c'}],
    [local_function, set(), {'y'}],
    [local_function_with_args, set(), {'y'}],
    [local_function_with_args_and_body,
     set(), {'y'}],
    [local_class, set(), {'y'}],
    [for_loop, {'z'}, {'y'}],
    [for_loop_many, set(), {'y'}],
    [for_loop_nested, set(), {'x'}],
    [for_loop_name_reference, set(), set()],
    [getitem_input, {'df'}, set()],
    [method_access_input, {'df'}, set()],
    [overriding_name, {'x', 'y'}, {'x', 'y'}],
],
                         ids=[
                             'only_outputs',
                             'simple',
                             'local_inputs',
                             'imports',
                             'imported_function',
                             'input_in_function_call',
                             'input_key_in_function_call',
                             'input_key_in_function_call_many',
                             'input_key_in_function_call_with_dot_access',
                             'modify_existing_getitem',
                             'modify_imported_getitem',
                             'built_in',
                             'built_in_as_arg',
                             'input_existing_object',
                             'define_multiple_outputs',
                             'local_function',
                             'local_function_with_args',
                             'local_function_with_args_and_body',
                             'local_class',
                             'for_loop',
                             'for_loop_many',
                             'for_loop_nested',
                             'for_loop_name_reference',
                             'getitem_input',
                             'method_access_input',
                             'overriding_name',
                         ])
def test_find_inputs_and_outputs(code_str, inputs, outputs):
    in_, out = static_analysis.find_inputs_and_outputs(code_str)

    assert in_ == inputs
    assert out == outputs


@pytest.mark.parametrize('snippets, ignore_input_names, expected', [
    [
        'import pandas as pd\n df = pd.read_csv("data.csv")', {'pd'},
        (set(), {'df'})
    ],
    [
        'import pandas as pd\nimport do_stuff\n'
        'df = do_stuff(pd.read_csv("data.csv"))', {'pd'}, (set(), {'df'})
    ],
],
                         ids=['simple', 'inside-function'])
def test_find_inputs_and_outputs_ignore_input_names(snippets,
                                                    ignore_input_names,
                                                    expected):
    assert static_analysis.find_inputs_and_outputs(
        snippets, ignore_input_names) == expected


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

# imports on its own section
imports = {
    'imports': 'import pandas as pd',
    'load': 'df = pd.read_csv("data.csv")',
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
            ({'y_test', 'X_test', 'y_train', 'X_train'}, {'lr', 'y_pred'}),
            'random-forest-regressor':
            ({'y_test', 'X_test', 'y_train', 'X_train'}, {'rf', 'y_pred'})
        },
    ],
    [imports, {
        'imports': (set(), set()),
        'load': (set(), {'df'})
    }],
],
                         ids=[
                             'eda',
                             'ml',
                             'imports',
                         ])
def test_find_io(snippets, expected):
    assert static_analysis.find_io(snippets) == expected


@pytest.mark.parametrize('io, expected', [
    [{
        'one': ({'a'}, {'b', 'c'}),
        'two': ({'b'}, set()),
    }, {
        'one': ({'a'}, {'b'}),
        'two': ({'b'}, set()),
    }],
])
def test_prune_io(io, expected):
    assert static_analysis.prune_io(io) == expected


@pytest.mark.parametrize('code, expected', [
    ['sns.histplot(df.some_column)', True],
    ['histplot(df.some_column)', True],
    ['sns.histplot(df)', True],
    ['histplot(df)', True],
    ['sns.histplot(df["key"])', True],
    ['def x(df):\n  pass', False],
    ['def x(df=1):\n  pass', False],
])
def test_inside_function_call(code, expected):
    leaf = get_first_leaf_with_value(code, 'df')
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


@pytest.mark.parametrize('snippets, names, a, b', [
    [{
        'a': 'import pandas',
        'b': 'import numpy as np'
    }, {
        'a': {'pandas'},
        'b': {'np'}
    },
     set(), {'pandas'}],
    [{
        'a': 'def x():\n    pass',
        'b': 'def y():\n    pass',
    }, {
        'a': {'x'},
        'b': {'y'}
    },
     set(), {'x'}],
])
def test_definitions_mapping(snippets, names, a, b):
    im = static_analysis.DefinitionsMapping(snippets)

    assert im._names == names
    assert im.get('a') == a
    assert im.get('b') == b


@pytest.mark.parametrize(
    'code, expected',
    [["""
def x():
    pass
""", {
        'x': '\ndef x():\n    pass'
    }], ["""
class X:
    pass
""", {
        'X': '\nclass X:\n    pass'
    }],
     [
         """
def x():
    pass

class X:
    pass
""", {
             'X': '\nclass X:\n    pass',
             'x': '\ndef x():\n    pass'
         }
     ]])
def test_find_defined_names_from_def_and_class(code, expected):
    out = (static_analysis.find_defined_names_from_def_and_class(
        parso.parse(code)))
    assert out == expected


@pytest.mark.parametrize('code, expected', [
    ['for x in range(10):\n    pass', {'x'}],
    ['for x, y in range(10):\n    pass', {'x', 'y'}],
    ['for x, (y, z) in range(10):\n    pass', {'x', 'y', 'z'}],
],
                         ids=[
                             'one',
                             'two',
                             'nested',
                         ])
def test_get_for_loop_defined_names(code, expected):
    tree = parso.parse(code)
    assert (static_analysis.find_for_loop_defined_names(
        tree.children[0]) == expected)


@pytest.mark.parametrize('code, expected', [
    ['for x in range(10):\n    pass', True],
    ['for x, y in range(10):\n    pass', True],
    ['for y, (z, x) in range(10):\n    pass', True],
    ['for y in range(10):\n    x = y + 1', False],
    ['for y in range(10):\n    z = y + 1\nfunction(x)', False],
],
                         ids=[
                             'single',
                             'tuple',
                             'nested',
                             'variable-in-loop-body',
                             'variable-in-loop-body-2',
                         ])
def test_for_loop_definition(code, expected):
    leaf = get_first_leaf_with_value(code, 'x')
    assert static_analysis.for_loop_definition(leaf) is expected


@pytest.mark.parametrize('code, expected', [
    ['name(x, y)', {'name', 'x', 'y'}],
    ['name(a=x, b=y)', {'name', 'x', 'y'}],
    ['name(x, b=y)', {'name', 'x', 'y'}],
    ['name({"x": x}, b=y)', {'name', 'x', 'y'}],
    ['name(x, b={"y": y})', {'name', 'x', 'y'}],
    ['name([x, y])', {'name', 'x', 'y'}],
],
                         ids=[
                             'simple',
                             'keywords',
                             'mixed',
                             'arg-dict',
                             'keyarg-dict',
                             'arg-list',
                         ])
def test_extract_inputs(code, expected):
    atom_exp = get_first_leaf_with_value(code, 'name').parent
    assert static_analysis.extract_inputs(atom_exp) == expected


@pytest.mark.parametrize('code, expected, index', [
    ['x = df.something', {'df'}, 0],
    ['x = df.something.another', {'df'}, 0],
    ['x = df.something()', {'df'}, 0],
    ['x = df["column"]', {'df'}, 0],
    ['x = df[another]', {'df', 'another'}, 0],
    ['x = df[function(another)]', {'df', 'function', 'another'}, 0],
    ['df = load()\nx = df + another', {'load'}, 0],
    ['x = y + z', {'y', 'z'}, 0],
    ['x = a + b + c + d', {'a', 'b', 'c', 'd'}, 0],
    ['x = [y for y in range(10)]', {'range'}, 0],
],
                         ids=[
                             'attribute',
                             'attribute-nested',
                             'method',
                             'getitem-literal',
                             'getitem-variable',
                             'getitem-nested',
                             'multiline',
                             'expression',
                             'expression-long',
                             'list-comprehension',
                         ])
def test_extract_inputs_with_atom_expr(code, expected, index):
    atom_exp = get_first_sibling_after_assignment(code, index=index)
    assert static_analysis.extract_inputs(atom_exp) == expected


# TODO: add nested list comprehension
@pytest.mark.parametrize('code, expected', [
    ['[x for x in range(10)]', {'range'}],
    ['(x for x in range(10))', {'range'}],
    ['[function(x) for x in range(10)]', {'range', 'function'}],
    ['[(x, y) for x, y in something(10)]', {'something'}],
],
                         ids=[
                             'left-expression',
                             'generator',
                             'both-expressions',
                             'many-variables',
                         ])
def test_get_inputs_in_list_comprehension(code, expected):
    tree = parso.parse(code)
    list_comp = tree.children[0].children[1]
    assert static_analysis.get_inputs_in_list_comprehension(
        list_comp) == expected


@pytest.mark.parametrize('code, expected', [
    ['[x for x in range(10)]', True],
    ['[x, y]', False],
],
                         ids=[
                             'for',
                             'list',
                         ])
def test_is_inside_list_comprehension(code, expected):
    node = get_first_leaf_with_value(code, 'x')
    assert static_analysis.is_inside_list_comprehension(node) is expected
