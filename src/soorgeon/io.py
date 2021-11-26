"""
Module to determine inputs and outputs from code snippets.
"""
from functools import reduce

import parso

from soorgeon import detect, get, definitions

_BUILTIN = set(__builtins__)


class ImportsParser:
    """Parses import statements to know which ones to inject to any given task

    Parameters
    ----------
    code_nb : str
        Notebook's source code
    """
    def __init__(self, code_nb):
        self._tree = parso.parse(code_nb)
        # maps defined names (from imports) to the source code
        self._name2code = definitions.from_imports(self._tree)

    def get_imports_cell_for_task(self, code_task):
        """
        Get the source code with the appropriate import statements for a task
        with the given code to work.
        """
        # NOTE: this was taken from jupyblog's source
        leaf = parso.parse(code_task).get_first_leaf()

        names = []

        while leaf:
            if leaf.type == 'name':
                names.append(leaf.value)

            leaf = leaf.get_next_leaf()

        imports = self._name2code

        # iterate over names defined by the imports and get the import
        # statement if content_subset uses it
        imports_to_use = []

        for name, import_code in imports.items():
            if name in names:
                imports_to_use.append(import_code)

        # remove duplicated elements but keep order, then join
        if imports:
            imports_to_use = ('\n'.join(list(dict.fromkeys(imports_to_use))) +
                              '\n\n\n').strip() or None
        else:
            imports_to_use = None

        # FIXME: once we parse the imports, we should remove them from the code
        # otherwise there are going to be duplicates

        return imports_to_use


def get_local_scope(leaf):
    """
    Returns a set of variables that are defined locally and thus should
    not be considered inputs (e.g., variables defined in a for loop)
    """
    parent = leaf.parent

    while parent:
        if parent.type == 'for_stmt':
            # call recursively for nested for loops to work
            return (find_for_loop_def_and_io(parent)[0]
                    | get_local_scope(parent.parent))

        # FIXME: this wont work with nested functions
        elif parent.type == 'funcdef':
            def_names = [
                c.get_defined_names() for c in parent.children[2].children
                if c.type == 'param'
            ]

            flatten = [name.value for sub in def_names for name in sub]

            return set(flatten)

        parent = parent.parent

    return set()


def find_for_loop_def_and_io(for_stmt, local_scope=None):
    """
    Return a set with the definitions and inputs a for loop. e.g.,
    for x, (y, z) in something() returns {'x', 'y', 'z'}, set()
    for i in range(input_) returns {'i'}, {'input_'}
    """
    # TODO: add a only_input flag for cases where we dont care about
    # parsin outputs
    if for_stmt.type != 'for_stmt':
        raise ValueError(f'Expected a node with type "for_stmt", '
                         f'got: {for_stmt} with type {for_stmt.type}')

    local_scope = local_scope or set()

    # get the parts that we need
    _, node_definition, _, node_iterator, _, body_node = for_stmt.children

    defined = extract_inputs(node_definition, parse_list_comprehension=False)
    iterator_in = extract_inputs(node_iterator, parse_list_comprehension=False)

    body_in, body_out = find_inputs_and_outputs_from_leaf(
        body_node.get_first_leaf(),
        ignore_input_names=defined,
        leaf_end=body_node.get_last_leaf())

    # Strictly speaking variables defined after the for keyword are also
    # outputs, since they're available after the loop ends (with the loop's
    # last value, however, we don't consider them here)
    return defined, (iterator_in | body_in) - local_scope, body_out


def find_function_scope_and_io(funcdef, local_scope=None):
    if funcdef.type != 'funcdef':
        raise ValueError(f'Expected a node with type "funcdef", '
                         f'got: {funcdef} with type {funcdef.type}')

    local_scope = local_scope or set()

    # get the parts that we need
    _, _, node_parameters, _, body_node = funcdef.children

    parameters = extract_inputs(node_parameters,
                                parse_list_comprehension=False)

    body_in, body_out = find_inputs_and_outputs_from_leaf(
        body_node.get_first_leaf(),
        ignore_input_names=parameters,
        leaf_end=body_node.get_last_leaf())

    return parameters, body_in - local_scope, body_out


def get_inputs_in_list_comprehension(node):
    if node.type != 'testlist_comp':
        raise ValueError('Expected node to have type '
                         f'"testlist_comp", got: {node.type}')

    compfor, synccompfor = node.children

    # parse the variables in the left expression
    # e.g., [expression(x) for x in range(10)]
    try:
        inputs_left = extract_inputs(compfor.children[0],
                                     parse_list_comprehension=False)
    except AttributeError:
        # if there isn't an expression but a single variable, we just
        # get the name e.g., [x for x in range(10)]
        inputs_left = {compfor.value}

    # these are the variables that the list comprehension declares
    declared = extract_inputs(synccompfor.children[1],
                              parse_list_comprehension=False)

    # parse the variables in the right expression
    # e,g, [x for x in expression(10)]
    # the expression part should be the element at index 3, note that this
    # is not the same as getting the last one because if the list comprehension
    # has an 'if' statement, that will be the last element
    inputs_right = extract_inputs(synccompfor.children[3],
                                  parse_list_comprehension=False)

    return (inputs_left | inputs_right) - declared


# TODO: this needs renaming, we are now using it to parse outputs as well
# see line 442
def extract_inputs(node,
                   parse_list_comprehension=True,
                   stop_at_end_of_list_comprehension=False,
                   only_getitem_and_attribute_access=False):
    """
    Extract inputs from an atomic expression
    e.g. function(x, y) returns {'function', 'x', 'y'}

    Parameters
    ----------
    parse_list_comprehension : bool, default=True
        Whether to parse any list comprehension if node is inside one
    """

    names = []

    leaf = node.get_first_leaf()
    # stop when you reach the end of the expression
    last = node.get_last_leaf()

    while leaf:
        if (parse_list_comprehension
                and detect.is_inside_list_comprehension(leaf)):
            list_comp = get.first_non_atom_expr_parent(leaf)
            inputs = get_inputs_in_list_comprehension(list_comp)
            names.extend(list(inputs))

            # skip to the end of the list comprehension
            leaf = list_comp.get_last_leaf()

            if stop_at_end_of_list_comprehension:
                break
        else:
            # is this a kwarg?
            try:
                key_arg = leaf.get_next_leaf().value == '='
            except Exception:
                key_arg = False

            # is this an attribute?
            try:
                is_attr = leaf.get_previous_leaf().value == '.'
            except Exception:
                is_attr = False

            if (leaf.type == 'name' and not key_arg and not is_attr
                    and leaf.value not in _BUILTIN):
                # not allowing reads, check that this is not geitem
                # or that is accessing an attribute in the next leaf
                try:
                    is_getitem = leaf.get_next_leaf().value == '['
                except Exception:
                    is_getitem = False

                try:
                    is_accessing_attr = leaf.get_next_leaf().value == '.'
                except Exception:
                    is_accessing_attr = False

                if (only_getitem_and_attribute_access
                        and (is_getitem or is_accessing_attr)):
                    names.append(leaf.value)
                elif not only_getitem_and_attribute_access:
                    names.append(leaf.value)

            if leaf is last:
                break

            leaf = leaf.get_next_leaf()

    return set(names)


def find_inputs_and_outputs(code_str, ignore_input_names=None):
    """
    Given a Python code string, find which variables the code consumes (not
    declared in the snipped) and which ones it exposes (declared in the
    snippet)

    Parameters
    ----------
    ignore_input_names : set
        Names that should not be considered inputs
    """
    tree = parso.parse(code_str)
    return find_inputs_and_outputs_from_tree(
        tree, ignore_input_names=ignore_input_names)


# FIXME: add a until_leaf parameter
def find_inputs_and_outputs_from_tree(tree, ignore_input_names=None):
    leaf = tree.get_first_leaf()
    # NOTE: we use this in find_inputs_and_outputs and ImportParser, maybe
    # move the functionality to a class so we only compute it once
    defined_names = set(definitions.from_imports(tree)) | set(
        definitions.from_def_and_class(tree))

    return find_inputs_and_outputs_from_leaf(
        leaf,
        ignore_input_names=ignore_input_names,
        defined_names=defined_names)


# FIXME: defined_names and ignore_inputs names should be only one arg
# maybe local_scope?
# FIXME: try nested functions, and also functions inside for loops and loops
# inside functions
def find_inputs_and_outputs_from_leaf(leaf,
                                      ignore_input_names=None,
                                      defined_names=None,
                                      leaf_end=None):
    ignore_input_names = ignore_input_names or set()
    defined_names = defined_names or set()

    inputs, outputs = [], set()

    local_variables = set()

    def clean_up_candidates(candidates, *others):
        # FIXME: this is not taking into account ignore_input_names
        # add a test to make it fail, then fix it
        candidates = candidates - set(_BUILTIN)
        candidates = candidates - set(defined_names)
        candidates = candidates - outputs

        for another in others:
            candidates = candidates - another

        return candidates

    while leaf:
        _inside_funcdef = detect.is_inside_funcdef(leaf)

        if not _inside_funcdef:
            local_variables = set()

        if detect.is_for_loop(leaf):
            # FIXME: i think is hould also pass the current foudn inputs
            # to local scope - write a test to break this
            (_, candidates_in, candidates_out) = find_for_loop_def_and_io(
                leaf.parent, local_scope=ignore_input_names)
            inputs.extend(clean_up_candidates(candidates_in, local_variables))
            outputs = outputs | candidates_out
            # jump to the end of the foor loop
            leaf = leaf.parent.get_last_leaf()
        elif detect.is_funcdef(leaf):
            # FIXME: i think is hould also pass the current foudn inputs
            # to local scope - write a test to break this
            (_, candidates_in, candidates_out) = find_function_scope_and_io(
                leaf.parent, local_scope=ignore_input_names)
            inputs.extend(clean_up_candidates(candidates_in, local_variables))
            outputs = outputs | candidates_out
            # jump to the end of the function definition loop
            leaf = leaf.parent.get_last_leaf()

        # the = operator is an indicator of [outputs] = [inputs]
        elif leaf.type == 'operator' and leaf.value == '=':
            next_s = leaf.get_next_sibling()
            previous = leaf.get_previous_leaf()

            # Process inputs
            inputs_current = extract_inputs(next_s)
            inputs_current = inputs_current - set(_BUILTIN)
            inputs_current = inputs_current - set(defined_names)

            # we dont call this anymore, we pass the local scope
            # as an arg
            # local = get_local_scope(leaf)
            local = set()

            for variable in inputs_current:
                # check if we're inside a for loop and ignore variables
                # defined there

                # only mark a variable as input if it hasn't been defined
                # locally
                if (variable not in outputs
                        and variable not in ignore_input_names
                        and variable not in local
                        and variable not in local_variables):
                    inputs.append(variable)

            # Process outputs

            # ignore keyword arguments, they aren't outputs
            # e.g. 'key' in something(key=value)
            # also ignore previous if modifying an existing object
            # e.g.,
            # a = {}
            # a['x'] = 1
            # a.b = 1

            if (previous.parent.type != 'argument'
                    and not _modifies_existing_object(leaf, outputs,
                                                      defined_names)):

                prev_sibling = leaf.get_previous_sibling()

                target = local_variables if _inside_funcdef else outputs

                # check if assigning multiple values
                # e.g., a, b = 1, 2 (testlist_star_expr)
                # [a, b] = 1, 2 (atom)
                # (a, b) = 1, 2 (atom)
                if prev_sibling.type in {'testlist_star_expr', 'atom'}:
                    target = target | set(
                        name.value
                        for name in prev_sibling.parent.get_defined_names())
                # nope, only one value
                elif prev_sibling.type == 'atom_expr':
                    target = target | extract_inputs(
                        prev_sibling, parse_list_comprehension=False)
                else:
                    target.add(previous.value)

                if _inside_funcdef:
                    local_variables = target
                else:
                    # before modifying the current outputs, check if there's
                    # anything on the left side of the = token that it's
                    # mutating a variable e.g.,:
                    # object.attribute = value
                    # or
                    # object['key'] = value
                    # this is the only case where something on the left side
                    # can be considered an input but only if the object hasn't
                    # been declared so far
                    inputs_candidates = extract_inputs(
                        prev_sibling,
                        parse_list_comprehension=False,
                        only_getitem_and_attribute_access=True)

                    # add to inputs if they haven't been locally defined
                    inputs_new = inputs_candidates - outputs
                    inputs.extend(inputs_new)

                    outputs = target

        # Process inputs scenario #2 - there is not '=' token but a function
        # call. variables inside function calls are inputs
        # e.g., some_function(df)
        # in this case, df is considered an input, except if it has been
        # locally defined.
        # e.g.,
        # df = something()
        # some_function(df)
        # FIXME: this is redundant because when we identify the '=' token, we
        # go to the first conditional, and the next leaf is the function call
        # so then we go into this conditional - we're skipping the left part
        # but not the right part of = yet
        elif (leaf.type == 'name' and (detect.is_inside_function_call(leaf)
                                       or detect.is_accessing_variable(leaf)
                                       or detect.is_inside_funcdef(leaf))
              # skip if this is to the left of an '=', because we'll check it
              # when we get to that token since it'll go to the first
              # conditional
              and not detect.is_left_side_of_assignment(leaf) and
              not detect.is_inside_list_comprehension(leaf) and
              leaf.value not in outputs and
              leaf.value not in ignore_input_names and
              leaf.value not in _BUILTIN and
              #   NO LONGER NEED TO CALL GET LOCAL SCOPE
              #   leaf.value not in get_local_scope(leaf) and
              leaf.value not in defined_names and leaf.value
              not in local_variables):
            inputs.extend(extract_inputs(leaf))
        elif leaf.type == 'name' and detect.is_inside_list_comprehension(leaf):
            inputs_new = extract_inputs(leaf,
                                        stop_at_end_of_list_comprehension=True)
            inputs.extend(inputs_new)

        next_s = leaf.get_next_sibling()

        # if we just parsed a list comprehension, skip until the end of it
        try:
            list_comp = next_s.children[1].type == 'testlist_comp'
        except (AttributeError, IndexError):
            list_comp = False

        if leaf_end and leaf == leaf_end:
            break

        if list_comp:
            leaf = next_s.get_last_leaf()
        else:
            leaf = leaf.get_next_leaf()

    return set(inputs), outputs


def _modifies_existing_object(leaf, outputs, names_from_imports):
    current = leaf.get_previous_sibling().get_first_leaf().value
    return current in outputs or current in names_from_imports


def _map_outputs(name, outputs):
    return [(out, name) for out in outputs]


def _get_upstream(name, inputs, providers):
    return [providers.get(input_, name) for input_ in inputs]


class ProviderMapping:
    """
    Determines which task produces a given variable to establish the upstream
    relationship

    Parameters
    ----------
    io : dict
        {task_name: (inputs, outputs), ...} mapping. Note that order is
        important, and is assumed that the order of the task_name keys matches
        the order of appearance of their corresponding cells in the notebook.
    """
    def __init__(self, io):
        self._io = io

    def _providers_for_task(self, name):
        """
        Returns a subset of io, only considering tasks that appear earlier
        in the notebook
        """
        out = {}

        for key, value in self._io.items():
            if key == name:
                break

            out[key] = value

        return _find_providers(out)

    def get(self, variable, task_name):
        """
        Return the provider of a certain variable for a task with name
        task_name. The function depends on task_name because if two tasks
        expose a variable with the same name, we need to resolve to the
        one closest to the task_name, by considering all previous sections
        in the notebook
        """
        providers = self._providers_for_task(task_name)

        provider = providers.get(variable)

        if not provider:
            raise KeyError('Could not find a task to '
                           f'obtain the {variable!r} that {task_name!r} uses')

        return provider


class DefinitionsMapping:
    """
    Returns the available names (from import statements, function and class
    definitions) available for a given
    snippet. We use this to determine which names should not be considered
    inputs in tasks, since they are module names
    """
    def __init__(self, snippets):
        self._names = {
            name: set(definitions.find_defined_names(parso.parse(code)))
            for name, code in snippets.items()
        }

    def get(self, name):
        out = set()

        for key, value in self._names.items():
            if key == name:
                break

            out = out | value

        return out


def find_upstream(snippets):
    """
    Parameters
    ----------
    snippets : dict
        {snippet_name: snippet, ...}
    """
    io = find_io(snippets)

    providers = ProviderMapping(io)

    # FIXME: this is going to generate duplicates if the task depends on >1
    # input from a given task, so we must remove duplicates
    upstream = {
        snippet_name: _get_upstream(snippet_name, v[0], providers)
        for snippet_name, v in io.items()
    }

    return upstream


def _find_providers(io):
    # variable -> snippet that defines variable mapping
    providers = [
        _map_outputs(snippet_name, v[1]) for snippet_name, v in io.items()
    ]

    providers = dict([i for sub in providers for i in sub])

    return providers


def find_io(snippets):
    """
    Generates a {snippet_name: (inputs, outputs), ...} mapping where inputs
    are the variables that snippet_name requires to work and outputs the ones
    that it creates
    """
    im = DefinitionsMapping(snippets)

    # FIXME: find_upstream already calls this, we should only compute it once
    io = {
        snippet_name:
        find_inputs_and_outputs(snippet,
                                ignore_input_names=im.get(snippet_name))
        for snippet_name, snippet in snippets.items()
    }

    return io


def prune_io(io):
    """
    Prunes an io mapping (as generated by find_io) that removes outputs that
    aren't used by any downstream task
    """
    # FIXME: order is important. if snippet at index i exports a, b, c, we
    # should only consider snippets beginning at index i + 1

    used = reduce(lambda x, y: x | y, (inputs for inputs, _ in io.values()))

    pruned = {}

    for key, (inputs, outputs) in io.items():
        outputs_pruned = outputs & used
        pruned[key] = (inputs, outputs_pruned)

    return pruned


def _leaf_iterator(tree):
    leaf = tree.get_first_leaf()

    while leaf:
        yield leaf

        leaf = leaf.get_next_leaf()


def remove_imports(code_str):
    """
    Remove all import statements from a code string
    """
    tree = parso.parse(code_str)

    to_remove = []

    for leaf in _leaf_iterator(tree):
        if leaf.parent.type in {'import_name', 'import_from'}:
            to_remove.append(leaf)

    for leaf in to_remove:
        leaf.parent.children = []

    return tree.get_code()