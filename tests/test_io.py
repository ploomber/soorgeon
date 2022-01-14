import testutils
from conftest import read_snippets

import parso
import pytest

from soorgeon import io


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

define_multiple_outputs_square_brackets = """
[a, b, c] = 1, 2, 3
"""

define_multiple_outputs_parenthesis = """
(a, b, c) = 1, 2, 3
"""

define_multiple_outputs_inside_function = """
import do_stuff

def fn():
    f, ax = do_stuff()
"""

define_multiple_replace_existing = """
b = 1

b, c = 2, 3

c.stuff()
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
    something = another + 1
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

for_loop_names_with_parenthesis = """
for a, (b, (c, d)) in range(10):
    x = a + b + c + d
"""

for_loop_nested = """
for i in range(10):
    for j in range(10):
        print(i + j)
"""

for_loop_nested_dependent = """
for filenames in ['file', 'name']:
    for char in filenames:
        print(char)
"""

for_loop_name_reference = """
for _, source in enumerate(10):
    some_function('%s' % source)
"""

for_loop_with_input = """
for range_ in range(some_input):
    pass
"""

for_loop_with_local_input = """
some_variable = 10

for range_ in range(some_variable):
    pass
"""

for_loop_with_input_attribute = """
for range_ in range(some_input.some_attribute):
    pass
"""

for_loop_with_input_nested_attribute = """
for range_ in range(some_input.some_attribute.another_attribute):
    pass
"""

for_loop_with_input_and_getitem = """
for range_ in range(some_input['some_key']):
    pass
"""

for_loop_with_input_and_getitem_input = """
for range_ in range(some_input[some_key]):
    pass
"""

for_loop_with_input_and_nested_getitem = """
for range_ in range(some_input[['some_key']]):
    pass
"""

for_loop_with_nested_input = """
for idx, range_ in enumerate(range(some_input)):
    pass
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

list_comprehension = """
[y for y in x]
"""

list_comprehension_attributes = """
[y.attribute for y in x.attribute]
"""

list_comprehension_with_conditional = """
targets = [1, 2, 3]
selected = [x for x in df.columns if x not in targets]
"""

list_comprehension_with_conditional_and_local_variable = """
import pandas as pd

df = pd.read_csv("data.csv")
features = [feature for feature in df.columns]
"""

list_comprehension_with_f_string = """
[f"'{s}'" for s in [] if s not in []]
"""

list_comprehension_with_f_string_assignment = """
y = [f"'{s}'" for s in [] if s not in []]
"""

list_comprehension_nested = """
out = [item for sublist in reduced_cats.values() for item in sublist]
"""

list_comprehension_nested_another = """
out = [[j for j in range(5)] for i in range(5)]
"""

list_comprehension_with_left_input = """
[x + y for x in range(10)]
"""

set_comprehension = """
output = {x for x in numbers if x % 2 == 0}
"""

dict_comprehension = """
output  = {x: y + 1 for x in numbers if x % 2 == 0}
"""

dict_comprehension_zip = """
output  = {x: y + 1 for x, z in zip(range(10), range(10)) if x % 2 == 0}
"""

function_with_global_variable = """
def some_function(a):
    return a + b
"""

# TODO: try with nested brackets like df[['something']]
# TODO: assign more than one at the same time df['a'], df['b'] = ...
mutating_input = """
df['new_column'] = df['some_column'] + 1
"""

# TODO: define inputs inside built-ins
# e.g.
# models = [a, b, c]
# models = {'a': a}

# TODO: we need a general function that finds the names after an =
# e.g. a = something(x=1, b=something)
# a = dict(a=1)
# b = {'a': x}

# this is an special case: since df hasn't been declared locally, it's
# considered an input even though it's on the left side of the = token,
# and it's also an output because it's modifying df
mutating_input_implicit = """
df['column'] = 1
"""

# counter example, local modification inside a function - that's ok
function_mutating_local_object = """
def fn():
    x = object()
    x['key'] = 1
    return x
"""

# add a case like failure but within a function
"""
def do(df):
    df['a'] = 1
"""

# there's also this problem if we mutatein a for loop
"""
# df becomes an output!
for col in df:
    col['x'] = col['x'] + 1
"""

# non-pure functions are problematic, too
"""
def do(df):
    df['a'] = 1


# here, df is an input that we should we from another task, but it should
# also be considered an output since we're mutating it, and, if the next
# task needs it, it'll need this version
do(df)
"""

nested_function_arg = """
import pd

pd.DataFrame({'key': y})
"""

nested_function_kwarg = """
import pd

pd.DataFrame(data={'key': y})
"""

# TODO: test nested context managers
context_manager = """
with open('file.txt') as f:
    x = f.read()
"""

f_string = """
f'{some_variable} {a_number:.2f} {an_object!r} {another!s}'
"""

f_string_assignment = """
s = f'{some_variable} {a_number:.2f} {an_object!r} {another!s}'
"""

class_ = """
class SomeClass:
    def __init__(self, param):
        self._param = param

    def some_method(self, a, b=0):
        return a + b

some_object = SomeClass(param=1)
"""


@pytest.mark.parametrize(
    'code_str, inputs, outputs', [
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
        [modify_imported_obj_getitem,
         set(), set()],
        [built_in, set(), {'mapping'}],
        [built_in_as_arg, set(), {'something'}],
        [input_existing_object, set(), {'X'}],
        [define_multiple_outputs,
         set(), {'a', 'b', 'c'}],
        [define_multiple_outputs_square_brackets,
         set(), {'a', 'b', 'c'}],
        [define_multiple_outputs_parenthesis,
         set(), {'a', 'b', 'c'}],
        [define_multiple_outputs_inside_function,
         set(), set()],
        [define_multiple_replace_existing,
         set(), {'b', 'c'}],
        [local_function, set(), {'y'}],
        [local_function_with_args, set(), {'y'}],
        [local_function_with_args_and_body,
         set(), {'y'}],
        [local_class, set(), {'y'}],
        [for_loop, {'z'}, {'y'}],
        [for_loop_many, set(), {'y'}],
        [for_loop_names_with_parenthesis,
         set(), {'x'}],
        [for_loop_nested, set(), set()],
        [for_loop_nested_dependent, set(),
         set()],
        [for_loop_name_reference, set(), set()],
        [for_loop_with_input, {'some_input'
                               }, set()],
        [for_loop_with_local_input,
         set(), {'some_variable'}],
        [for_loop_with_input_attribute,
         {'some_input'}, set()],
        [for_loop_with_input_nested_attribute,
         {'some_input'
          }, set()],
        [for_loop_with_input_and_getitem,
         {'some_input'
          }, set()],
        [
            for_loop_with_input_and_getitem_input, {'some_input', 'some_key'},
            set()
        ],
        [for_loop_with_input_and_nested_getitem,
         {'some_input'
          }, set()],
        [for_loop_with_nested_input,
         {'some_input'}, set()],
        [getitem_input, {'df'}, set()],
        [method_access_input, {'df'}, set()],
        [overriding_name, {'x', 'y'}, {'x', 'y'}],
        [list_comprehension, {'x'}, set()],
        [list_comprehension_attributes,
         {'x'}, set()],
        [list_comprehension_with_conditional, {'df'}, {'selected', 'targets'}],
        [
            list_comprehension_with_conditional_and_local_variable,
            set(), {'df', 'features'}
        ],
        [list_comprehension_with_f_string,
         set(), set()],
        [list_comprehension_with_f_string_assignment,
         set(), {'y'}],
        [list_comprehension_nested, {'reduced_cats'}, {'out'}],
        [list_comprehension_nested_another,
         set(), {'out'}],
        [list_comprehension_with_left_input,
         {'y'}, set()],
        [set_comprehension, {'numbers'}, {'output'}],
        [dict_comprehension, {'numbers', 'y'}, {'output'}],
        [dict_comprehension_zip, {'y'}, {'output'}],
        [function_with_global_variable,
         {'b'}, set()],
        [mutating_input, {'df'}, {'df'}],
        [mutating_input_implicit, {'df'}, {'df'}],
        [function_mutating_local_object,
         set(), set()],
        [nested_function_arg, {'y'}, set()],
        [nested_function_kwarg, {'y'}, set()],
        [context_manager, set(), {'x'}],
        [
            f_string, {'some_variable', 'a_number', 'an_object', 'another'},
            set()
        ],
        [
            f_string_assignment,
            {'some_variable', 'a_number', 'an_object', 'another'}, {'s'}
        ],
        [class_, set(), {'some_object'}],
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
        'define_multiple_outputs_square_brackets',
        'define_multiple_outputs_parenthesis',
        'define_multiple_outputs_inside_function',
        'define_multiple_replace_existing',
        'local_function',
        'local_function_with_args',
        'local_function_with_args_and_body',
        'local_class',
        'for_loop',
        'for_loop_many',
        'for_loop_names_with_parenthesis',
        'for_loop_nested',
        'for_loop_nested_dependent',
        'for_loop_name_reference',
        'for_loop_with_input',
        'for_loop_with_local_input',
        'for_loop_with_input_attribute',
        'for_loop_with_input_nested_attribute',
        'for_loop_with_input_and_getitem',
        'for_loop_with_input_and_getitem_input',
        'for_loop_with_input_and_nested_getitem',
        'for_loop_with_nested_input',
        'getitem_input',
        'method_access_input',
        'overriding_name',
        'list_comprehension',
        'list_comprehension_attributes',
        'list_comprehension_with_conditional',
        'list_comprehension_with_conditional_and_local_variable',
        'list_comprehension_with_f_string',
        'list_comprehension_with_f_string_assignment',
        'list_comprehension_nested',
        'list_comprehension_nested_another',
        'list_comprehension_with_left_input',
        'set_comprehension',
        'dict_comprehension',
        'dict_comprehension_zip',
        'function_with_global_variable',
        'mutating_input',
        'mutating_input_implicit',
        'function_mutating_local_object',
        'nested_function_arg',
        'nested_function_kwarg',
        'context_manager',
        'f_string',
        'f_string_assignment',
        'class_',
    ])
def test_find_inputs_and_outputs(code_str, inputs, outputs):
    in_, out = io.find_inputs_and_outputs(code_str)

    assert in_ == inputs
    assert out == outputs


@pytest.mark.parametrize('code, expected_len', [
    ['[x for x in range(10)]', 1],
    ['[i for row in matrix for i in row]', 2],
    ['[i for matrix in tensor for row in matrix for i in row]', 3],
],
                         ids=[
                             'simple',
                             'nested',
                             'nested-nested',
                         ])
def test_flatten_sync_comp_for(code, expected_len):
    synccompfor = parso.parse(code).children[0].children[1].children[1]

    assert len(io._flatten_sync_comp_for(synccompfor)) == expected_len


@pytest.mark.parametrize('code, in_expected, declared_expected', [
    ['[x for x in range(10)]', set(), {'x'}],
],
                         ids=[
                             'simple',
                         ])
def test_find_sync_comp_for_inputs_and_scope(code, in_expected,
                                             declared_expected):
    synccompfor = parso.parse(code).children[0].children[1].children[1]

    in_, declared = io._find_sync_comp_for_inputs_and_scope(synccompfor)

    assert in_ == in_expected
    assert declared == declared_expected


@pytest.mark.parametrize('snippets, local_scope, expected', [
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
def test_find_inputs_and_outputs_local_scope(snippets, local_scope, expected):
    assert io.find_inputs_and_outputs(snippets, local_scope) == expected


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
    m = io.ProviderMapping(io.find_io(eda))

    assert m._providers_for_task('load') == {}
    assert m._providers_for_task('clean') == {'df': 'load'}
    assert m._providers_for_task('plot') == {'df': 'clean'}
    assert m.get('df', 'clean') == 'load'


def test_providermapping_error():
    m = io.ProviderMapping(io.find_io(eda))

    with pytest.raises(KeyError) as excinfo:
        m.get('unknown_variable', 'clean')

    expected = ("Could not find a task to obtain the "
                "'unknown_variable' that 'clean' uses")

    assert expected in str(excinfo.value)


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
    assert io.find_upstream(snippets) == expected


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
    assert io.find_io(snippets) == expected


@pytest.mark.parametrize('io_, expected', [
    [{
        'one': ({'a'}, {'b', 'c'}),
        'two': ({'b'}, set()),
    }, {
        'one': ({'a'}, {'b'}),
        'two': ({'b'}, set()),
    }],
])
def test_prune_io(io_, expected):
    assert io.prune_io(io_) == expected


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
    ip = io.ImportsParser(code_nb)
    assert ip.get_imports_cell_for_task(code_task) == expected


@pytest.mark.parametrize('code, expected', [
    ['import pandas as pd\nimport numpy as np', '\n'],
    ['import math', ''],
    ['import pandas as pd\n1+1', '\n1+1'],
    ['import math\n1+1', '\n1+1'],
])
def test_remove_imports(code, expected):
    assert io.remove_imports(code) == expected


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
    im = io.DefinitionsMapping(snippets)

    assert im._names == names
    assert im.get('a') == a
    assert im.get('b') == b


@pytest.mark.parametrize('code, def_expected, in_expected, out_expected', [
    ['for x in range(10):\n    pass', {'x'},
     set(), set()],
    ['for x, y in range(10):\n    pass', {'x', 'y'},
     set(), set()],
    ['for x, (y, z) in range(10):\n    pass', {'x', 'y', 'z'},
     set(),
     set()],
    ['for x in range(10):\n    pass\n\nj = i', {'x'},
     set(), set()],
    [
        'for i, a_range in enumerate(range(x)):\n    pass', {'i', 'a_range'},
        {'x'},
        set()
    ],
    ['for i in range(10):\n    print(i + 10)', {'i'},
     set(), set()],
],
                         ids=[
                             'one',
                             'two',
                             'nested',
                             'code-outside-for-loop',
                             'nested-calls',
                             'uses-local-sope-in-body',
                         ])
def test_find_for_loop_def_and_io(code, def_expected, in_expected,
                                  out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = io.find_for_loop_def_and_io(tree.children[0])
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize('code, def_expected, in_expected, out_expected', [
    ['with open("file") as f:\n    pass', {'f'},
     set(), set()],
    ['with open("file"):\n    pass',
     set(), set(), set()],
    [
        'with open("file") as f, open("another") as g:\n    pass', {'f', 'g'},
        set(),
        set()
    ],
    ['with open("file") as f:\n    x = f.read()', {'f'},
     set(), {'x'}],
    ['with open("file") as f:\n    x, y = f.read()', {'f'},
     set(), {'x', 'y'}],
    [
        'with open(some_path) as f:\n    x = f.read()', {'f'}, {'some_path'},
        {'x'}
    ],
    [
        'with open(some_path, another_path) as (f, ff):\n    x = f.read()',
        {'f', 'ff'}, {'some_path', 'another_path'}, {'x'}
    ],
],
                         ids=[
                             'one',
                             'no-alias',
                             'two',
                             'output-one',
                             'output-many',
                             'input-one',
                             'input-many',
                         ])
def test_find_context_manager_def_and_io(code, def_expected, in_expected,
                                         out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = io.find_context_manager_def_and_io(tree.children[0])
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize('code, def_expected, in_expected, out_expected', [
    ['def fn(x):\n    pass', {'x'},
     set(), set()],
    ['def fn(x, y):\n    pass', {'x', 'y'},
     set(), set()],
    ['def fn(x, y):\n    something = z + x + y', {'x', 'y'}, {'z'},
     set()],
    ['def fn(x, y):\n    z(x, y)', {'x', 'y'}, {'z'},
     set()],
    ['def fn(x, y):\n    z.do(x, y)', {'x', 'y'}, {'z'},
     set()],
    ['def fn(x, y):\n    z[x]', {'x', 'y'}, {'z'},
     set()],
    ['def fn(x, y):\n    z + x + y', {'x', 'y'}, {'z'},
     set()],
    ['def fn(x, y):\n    z', {'x', 'y'}, {'z'},
     set()],
    ['def fn() -> Mapping[str, int]:\n    pass',
     set(), {'Mapping'},
     set()],
    [
        'def fn(x: int, y: Mapping[str, int]):\n    z + x + y',
        {'Mapping', 'x', 'y'}, {'z'},
        set()
    ],
],
                         ids=[
                             'arg-one',
                             'arg-two',
                             'uses-outer-scope',
                             'uses-outer-scope-callable',
                             'uses-outer-scope-attribute',
                             'uses-outer-scope-getitem',
                             'uses-outer-scope-no-assignment',
                             'uses-outer-scope-reference',
                             'annotation-return',
                             'annotation-args',
                         ])
def test_find_function_scope_and_io(code, def_expected, in_expected,
                                    out_expected):
    tree = parso.parse(code)
    # TODO: test with non-empty local_scope parameter
    def_, in_, out = (io.find_function_scope_and_io(tree.children[0]))
    assert def_ == def_expected
    assert in_ == in_expected
    assert out == out_expected


@pytest.mark.parametrize(
    'code, expected', [['name(x, y)', {'name', 'x', 'y'}],
                       ['name(a=x, b=y)', {'name', 'x', 'y'}],
                       ['name(x, b=y)', {'name', 'x', 'y'}],
                       ['name({"x": x}, b=y)', {'name', 'x', 'y'}],
                       ['name(x, b={"y": y})', {'name', 'x', 'y'}],
                       ['name([x, y])', {'name', 'x', 'y'}],
                       ['name["a"]', {'name'}], ['name.atribute', {'name'}]],
    ids=[
        'simple',
        'keywords',
        'mixed',
        'arg-dict',
        'keyarg-dict',
        'arg-list',
        'getitem',
        'attribute',
    ])
def test_find_inputs(code, expected):
    atom_exp = testutils.get_first_leaf_with_value(code, 'name').parent
    assert io.find_inputs(atom_exp) == expected


@pytest.mark.parametrize('code, expected', [
    ['name["a"]', {'name'}],
    ['name.atribute', {'name'}],
    ['name', set()],
],
                         ids=[
                             'getitem',
                             'attribute',
                             'name',
                         ])
def test_find_inputs_only_getitem_and_attribute_access(code, expected):
    atom_exp = testutils.get_first_leaf_with_value(code, 'name').parent
    out = io.find_inputs(atom_exp, only_getitem_and_attribute_access=True)
    assert out == expected


@pytest.mark.parametrize('code, expected', [
    ['[x for x in df["some_key"]]', {'df'}],
    ['[x for x in df["some_key"]["another_key"]]', {'df'}],
],
                         ids=[
                             'getitem',
                             'getitem-nested',
                         ])
def test_find_inputs_only_getitem_and_attribute_access_list_comprehension(
        code, expected):
    out = io.find_inputs(parso.parse(code),
                         only_getitem_and_attribute_access=True)
    assert out == expected


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
    ['x = [y for y in range(10)]', set(), 0],
    ['x = np.std([y for y in range(10)])', {'np'}, 0],
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
                             'list-comprehension-as-arg',
                         ])
def test_find_inputs_with_atom_expr(code, expected, index):
    atom_exp = get_first_sibling_after_assignment(code, index=index)
    assert io.find_inputs(atom_exp) == expected


@pytest.mark.parametrize('code, expected', [
    ['[x for x in range(10)]', set()],
    ['[f"{x}" for x in range(10)]', set()],
    ['(x for x in range(10))', set()],
    ['[function(x) for x in range(10)]', {'function'}],
    ['[(x, y) for x, y in something(10)]', {'something'}],
    ['[x.attribute for x in range(10)]',
     set()],
    ['[x for x in obj if x > 0]', {'obj'}],
    ['[i for row in matrix for i in row]', {'matrix'}],
    ['[i for matrix in tensor for row in matrix for i in row]', {'tensor'}],
],
                         ids=[
                             'left-expression',
                             'f-string',
                             'generator',
                             'both-expressions',
                             'many-variables',
                             'attributes',
                             'conditional',
                             'nested',
                             'nested-double',
                         ])
def test_find_list_comprehension_inputs(code, expected):
    tree = parso.parse(code)
    list_comp = tree.children[0].children[1]
    assert io.find_comprehension_inputs(list_comp) == expected


@pytest.mark.parametrize('code, expected', [
    ['for i in range(10):\n    y = x + i', {'i'}],
    ['for i, j in something():\n    y = x + i', {'i', 'j'}],
    ['def function(i):\n    y = x + i', {'i'}],
    ['def function(i, j):\n    y = x + i + j', {'i', 'j'}],
],
                         ids=[
                             'for',
                             'for-many',
                             'def',
                             'def-many',
                         ])
def test_get_local_scope(code, expected):
    node = testutils.get_first_leaf_with_value(code, 'x')
    assert io.get_local_scope(node) == expected


# TODO: try nested
# @pytest.mark.parametrize('code, expected', [
#     ['a = 1', False],
#     ['a, b = 1, 2', False],
#     ['existing = 1', True],
#     ['a, existing = 1, 2', True],
#     ['existing, b = 1, 2', True],
#     ['(a, existing) = 1, 2', True],
#     ['(existing, b) = 1, 2', True],
#     ['[a, existing] = 1, 2', True],
#     ['[existing, b] = 1, 2', True],
# ])
# def test_modifies_existing_object(code, expected):
#     leaf = testutils.get_first_leaf_with_value(code, '=')
#     assert io._modifies_existing_object(leaf, {'existing'}, set()) is
# expected
