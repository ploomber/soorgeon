import pytest

from soorgeon import io

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

local_function_with_kwargs = """
def my_function(a, b, c=None):
    return a + b + c

y = my_function(1, 2, 3)
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

list_comprehension_nested_more = """
out = [[[k for k in range(j)] for j in range(i)] for i in range(5)]
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

lambda_ = """
lambda x: x
"""

lambda_with_input = """
lambda x: x + y
"""

lambda_as_arg = """
import something
something(1, lambda x: x)
"""

lambda_assignment = """
out = lambda x: x
"""

lambda_with_input_assignment = """
out = lambda x: x + y
"""

lambda_as_arg_assignment = """
import something
out = something(1, lambda x: x)
"""


@pytest.mark.parametrize(
    "code_str, inputs, outputs",
    [
        [only_outputs, set(), {"x", "y"}],
        [simple, {"x", "y"}, {"z"}],
        [local_inputs, set(), {"x", "y", "z"}],
        [imports, set(), {"z"}],
        [imported_function, set(), {"df"}],
        [input_in_function_call, {"df"}, set()],
        [input_key_in_function_call, {"df"}, set()],
        [input_key_in_function_call_many, {"df", "df_another"}, set()],
        [input_key_in_function_call_with_dot_access, {"df"}, set()],
        [modify_existing_obj_getitem, set(), {"mapping"}],
        [modify_imported_obj_getitem, set(), set()],
        [built_in, set(), {"mapping"}],
        [built_in_as_arg, set(), {"something"}],
        [input_existing_object, set(), {"X"}],
        [define_multiple_outputs, set(), {"a", "b", "c"}],
        [define_multiple_outputs_square_brackets, set(), {"a", "b", "c"}],
        [define_multiple_outputs_parenthesis, set(), {"a", "b", "c"}],
        [define_multiple_outputs_inside_function, set(), set()],
        [
            define_multiple_replace_existing,
            set(),
            {"b", "c"},
        ],
        [local_function, set(), {"y"}],
        [local_function_with_args, set(), {"y"}],
        [
            local_function_with_args_and_body,
            set(),
            {"y"},
        ],
        [
            local_function_with_kwargs,
            set(),
            {"y"},
        ],
        [local_class, set(), {"y"}],
        [for_loop, {"z"}, {"y"}],
        [for_loop_many, set(), {"y"}],
        [for_loop_names_with_parenthesis, set(), {"x"}],
        [for_loop_nested, set(), set()],
        [for_loop_nested_dependent, set(), set()],
        [for_loop_name_reference, set(), set()],
        [for_loop_with_input, {"some_input"}, set()],
        [for_loop_with_local_input, set(), {"some_variable"}],
        [for_loop_with_input_attribute, {"some_input"}, set()],
        [for_loop_with_input_nested_attribute, {"some_input"}, set()],
        [for_loop_with_input_and_getitem, {"some_input"}, set()],
        [for_loop_with_input_and_getitem_input, {"some_input", "some_key"}, set()],
        [for_loop_with_input_and_nested_getitem, {"some_input"}, set()],
        [for_loop_with_nested_input, {"some_input"}, set()],
        [getitem_input, {"df"}, set()],
        [method_access_input, {"df"}, set()],
        [overriding_name, {"x", "y"}, {"x", "y"}],
        [list_comprehension, {"x"}, set()],
        [list_comprehension_attributes, {"x"}, set()],
        [list_comprehension_with_conditional, {"df"}, {"selected", "targets"}],
        [
            list_comprehension_with_conditional_and_local_variable,
            set(),
            {"df", "features"},
        ],
        [list_comprehension_with_f_string, set(), set()],
        [list_comprehension_with_f_string_assignment, set(), {"y"}],
        [list_comprehension_nested, {"reduced_cats"}, {"out"}],
        [
            list_comprehension_nested_another,
            set(),
            {"out"},
        ],
        [
            list_comprehension_nested_more,
            set(),
            {"out"},
        ],
        [
            list_comprehension_with_left_input,
            {"y"},
            set(),
        ],
        [set_comprehension, {"numbers"}, {"output"}],
        [dict_comprehension, {"numbers", "y"}, {"output"}],
        [dict_comprehension_zip, {"y"}, {"output"}],
        [function_with_global_variable, {"b"}, set()],
        [mutating_input, {"df"}, {"df"}],
        [mutating_input_implicit, {"df"}, {"df"}],
        [function_mutating_local_object, set(), set()],
        [nested_function_arg, {"y"}, set()],
        [nested_function_kwarg, {"y"}, set()],
        [context_manager, set(), {"x"}],
        [f_string, {"some_variable", "a_number", "an_object", "another"}, set()],
        [
            f_string_assignment,
            {"some_variable", "a_number", "an_object", "another"},
            {"s"},
        ],
        [class_, set(), {"some_object"}],
        [lambda_, set(), set()],
        [lambda_with_input, {"y"}, set()],
        [lambda_as_arg, set(), set()],
        [lambda_assignment, set(), {"out"}],
        [lambda_with_input_assignment, {"y"}, {"out"}],
        [lambda_as_arg_assignment, set(), {"out"}],
    ],
    ids=[
        "only_outputs",
        "simple",
        "local_inputs",
        "imports",
        "imported_function",
        "input_in_function_call",
        "input_key_in_function_call",
        "input_key_in_function_call_many",
        "input_key_in_function_call_with_dot_access",
        "modify_existing_getitem",
        "modify_imported_getitem",
        "built_in",
        "built_in_as_arg",
        "input_existing_object",
        "define_multiple_outputs",
        "define_multiple_outputs_square_brackets",
        "define_multiple_outputs_parenthesis",
        "define_multiple_outputs_inside_function",
        "define_multiple_replace_existing",
        "local_function",
        "local_function_with_args",
        "local_function_with_args_and_body",
        "local_function_with_kwargs",
        "local_class",
        "for_loop",
        "for_loop_many",
        "for_loop_names_with_parenthesis",
        "for_loop_nested",
        "for_loop_nested_dependent",
        "for_loop_name_reference",
        "for_loop_with_input",
        "for_loop_with_local_input",
        "for_loop_with_input_attribute",
        "for_loop_with_input_nested_attribute",
        "for_loop_with_input_and_getitem",
        "for_loop_with_input_and_getitem_input",
        "for_loop_with_input_and_nested_getitem",
        "for_loop_with_nested_input",
        "getitem_input",
        "method_access_input",
        "overriding_name",
        "list_comprehension",
        "list_comprehension_attributes",
        "list_comprehension_with_conditional",
        "list_comprehension_with_conditional_and_local_variable",
        "list_comprehension_with_f_string",
        "list_comprehension_with_f_string_assignment",
        "list_comprehension_nested",
        "list_comprehension_nested_another",
        "list_comprehension_nested_more",
        "list_comprehension_with_left_input",
        "set_comprehension",
        "dict_comprehension",
        "dict_comprehension_zip",
        "function_with_global_variable",
        "mutating_input",
        "mutating_input_implicit",
        "function_mutating_local_object",
        "nested_function_arg",
        "nested_function_kwarg",
        "context_manager",
        "f_string",
        "f_string_assignment",
        "class_",
        "lambda_",
        "lambda_with_input",
        "lambda_as_arg",
        "lambda_assignment",
        "lambda_with_input_assignment",
        "lambda_as_arg_assignment",
    ],
)
def test_find_inputs_and_outputs(code_str, inputs, outputs):
    in_, out = io.find_inputs_and_outputs(code_str)

    assert in_ == inputs
    assert out == outputs
