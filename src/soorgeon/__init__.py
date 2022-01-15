"""
We want to parse a series of Python snippets (one per notebook section -
splitted by H2 headers). For each snippet, we need to know which variables
they need (from previous sections) and which ones they expose (available for
upcoming sections). There are a few prime cases we want to cover.

1. Assignment: a = 1
2. Function calling: function(1, b=2)
3. If/While statements
4. List comprehensions
5. For loops
6. Function definitions

The first three are simple, however list comprehensions, for loops and
functions introduce a new complexity: they can be nested. Example:

for i in range(10):
    for j in range(10):
        print(i, j)

Hence, we need to keep into account that they define a local scope, hence
some of the variables should not be considered exposed to upcoming tasks.
For example, in our nested loop i and j are not considered exposed.
if/while statements can also be arbitrarily nested but they do not define a
local scope, so we don't have that complication.

To deal with this possibly nested logic, we have a function
(find_inputs_and_outputs_from_tree) that goes through every leaf in the ast,
and determines which sub function to execute. For example, if it finds a 'for'
keyword, it calls find_for_loop_def_and_io, which parses a for
structure.

Note that find_inputs_and_outputs_from_tree and
find_for_loop_def_and_io may call find_inputs_and_outputs_from_tree
again, because such a function deals with arbitrary structures. However,
when doing nested calls to find_inputs_and_outputs_from_tree, one should
set the leaf parameter to tell the function when to stop and return.

The second type of functions are detectors, which determine which
sub-function to call (e.g., is_for_loop identifies if we're about to
enter a for loop). These detectors should return False if we're not inside
the structure they identify, if we're in a given structure, they should return
the last leaf, so we can pass this to the parsing function in the until leaf
argument.
"""

__version__ = '0.0.7'
