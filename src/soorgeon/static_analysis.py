from functools import reduce

import parso

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
        self._name2code = find_defined_names_from_imports(self._tree)

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


# NOTE: we use this in find_inputs_and_outputs and ImportParser, maybe
# move the functionality to a class so we only compute it once
def find_defined_names_from_imports(tree):
    # build a defined-name -> import-statement-code mapping. Note that
    # the same code may appear more than once if it defines more than one name
    # e.g. from package import a, b, c
    imports = [{
        name.value: import_.get_code().rstrip()
        for name in import_.get_defined_names()
    } for import_ in tree.iter_imports()]

    if imports:
        imports = reduce(lambda x, y: {**x, **y}, imports)
    else:
        imports = {}

    return imports


def find_defined_names_from_def_and_class(tree):
    fns = {
        fn.name.value: fn.get_code().rstrip()
        for fn in tree.iter_funcdefs()
    }

    classes = {
        class_.name.value: class_.get_code().rstrip()
        for class_ in tree.iter_classdefs()
    }

    return {**fns, **classes}


def find_defined_names(tree):
    return {
        **find_defined_names_from_imports(tree),
        **find_defined_names_from_def_and_class(tree)
    }


def inside_function_call(leaf):
    # ignore it if this is a function definition
    if leaf.parent.type == 'param':
        return False

    next_sibling = leaf.get_next_sibling()

    try:
        next_sibling_value = next_sibling.value
    except AttributeError:
        next_sibling_value = None

    # ignore names in keyword arguments
    # e.g., some_function(x=1)
    # (x does not count since it's)
    if next_sibling_value == '=':
        return False

    # first case covers something like: function(df)
    # second case cases like: function(df.something)
    # NOTE: do we need more checks to ensure we're in the second case?
    # maybe check if we have an actual dot, or we're using something like
    # df[key]?
    return inside_parenthesis(leaf) or inside_parenthesis(leaf.parent)


def inside_parenthesis(node):
    try:
        left = node.get_previous_sibling().value == '('
    except AttributeError:
        left = False

    try:
        right = node.get_next_sibling().value == ')'
    except AttributeError:
        right = False

    return left and right


def inside_funcdef(leaf):
    parent = leaf.parent

    while parent:
        if parent.type == 'funcdef':
            return True

        parent = parent.parent

    return False


def get_local_scope(leaf):
    """
    Returns a set of variables that are defined locally and thus should
    not be considered inputs (e.g., variables defined in a for loop)
    """
    # FIXME: this wont work with nested for loops
    parent = leaf.parent

    while parent:
        if parent.type == 'for_stmt':
            return find_for_loop_defined_names(parent)

        parent = parent.parent

    return set()


def find_for_loop_defined_names(for_stmt):
    """
    Return a set with the variables defined in a for loop. e.g.,
    for x, (y, z) in .... returns {'x', 'y', 'z'}
    """
    if for_stmt.type != 'for_stmt':
        raise ValueError(f'Expected a node with type "for_stmt", '
                         f'got: {for_stmt} with type {for_stmt.type}')

    variables = for_stmt.children[1]

    names = []

    leaf = variables.get_first_leaf()

    while leaf:
        if leaf.type == 'keyword' and leaf.value == 'in':
            break

        if leaf.type == 'name':
            names.append(leaf.value)

        leaf = leaf.get_next_leaf()

    return set(names)


def for_loop_definition(leaf):
    has_suite_parent = False
    parent = leaf.parent

    while parent:
        if parent.type == 'suite':
            has_suite_parent = True

        if parent.type == 'for_stmt':
            return not has_suite_parent

        parent = parent.parent

    return False


def accessing_variable(leaf):
    """
    For a given node of type name, determine if it's used
    """
    # NOTE: what if we only have the name and we are not doing anything?
    # like:
    # df
    # that still counts as dependency
    try:
        children = leaf.get_next_sibling().children
    except Exception:
        return False

    getitem = children[0].value == '[' and children[-1].value == ']'
    dotaccess = children[0].value == '.'
    # FIXME: adding dotacess breaks other tests
    return getitem or dotaccess


def is_inside_list_comprehension(node):
    return (node.parent.type == 'testlist_comp'
            and node.parent.children[1].type == 'sync_comp_for')


def get_inputs_in_list_comprehension(node):
    if node.type != 'testlist_comp':
        raise ValueError('Expected node to have type '
                         f'"testlist_comp", got: {node.type}')

    compfor, synccompfor = node.children

    # parse the variables in the left expression
    # e.g., [expression(x) for x in range(10)]
    try:
        expression_left = extract_inputs(compfor.children[0])
    except AttributeError:
        # if there isn't an expression but a single variable, we just
        # get the name e.g., [x for x in range(10)]
        expression_left = {compfor.value}

    # this are the variables that the list comprehension declares
    declared = extract_inputs(synccompfor.children[1])

    # parse the variables in the right expression
    # e,g, [x for x in expression(10)]
    expression_right = extract_inputs(synccompfor.children[-1])

    return (expression_left | expression_right) - declared


def extract_inputs(node):
    """
    Extract inputs from an atomic expression
    e.g. function(x, y) returns {'function', 'x', 'y'}
    """
    names = []

    leaf = node.get_first_leaf()
    # stop when you reach the end of the expression
    last = node.get_last_leaf()

    while leaf:
        if is_inside_list_comprehension(leaf):
            names.extend(list(get_inputs_in_list_comprehension(leaf.parent)))

            # skip to the end of the list comprehension
            leaf = leaf.parent.get_last_leaf()
        else:
            # is this a kwarg?
            try:
                key_arg = leaf.get_next_leaf().value == '='
            except Exception:
                key_arg = False

            # attribute access?
            try:
                attr_access = leaf.get_previous_leaf().value == '.'
            except Exception:
                attr_access = False

            if leaf.type == 'name' and not key_arg and not attr_access:
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

    ignore_input_names = ignore_input_names or set()
    tree = parso.parse(code_str)
    leaf = tree.get_first_leaf()

    # NOTE: we use this in find_inputs_and_outputs and ImportParser, maybe
    # move the functionality to a class so we only compute it once
    defined_names = set(find_defined_names_from_imports(tree)) | set(
        find_defined_names_from_def_and_class(tree))

    inputs, outputs = [], set()

    while leaf:
        # FIXME: is there a more efficient way to do this? if you find a
        # funcspec, jump to the end of it
        if inside_funcdef(leaf) or for_loop_definition(leaf):
            pass

        # the = operator is an indicator of [outputs] = [inputs]
        elif leaf.type == 'operator' and leaf.value == '=':
            next_s = leaf.get_next_sibling()
            previous = leaf.get_previous_leaf()

            # Process inputs
            inputs_current = extract_inputs(next_s)
            inputs_current = inputs_current - set(_BUILTIN)
            inputs_current = inputs_current - set(defined_names)

            local = get_local_scope(leaf)

            for variable in inputs_current:
                # check if we're inside a for loop and ignore variables
                # defined there

                # only mark a variable as input if it hasn't been defined
                # locally
                if (variable not in outputs
                        and variable not in ignore_input_names
                        and variable not in local):
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

                # check if assigning multiple values
                # e.g., a, b = 1, 2
                if prev_sibling.type == 'testlist_star_expr':
                    outputs = outputs | set(
                        name.value
                        for name in prev_sibling.parent.get_defined_names())
                # nope, only one value
                else:
                    outputs.add(previous.value)

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
        # so then we go into this conditional
        elif (leaf.type == 'name'
              and (inside_function_call(leaf) or accessing_variable(leaf))
              and leaf.value not in outputs
              and leaf.value not in ignore_input_names
              and leaf.value not in _BUILTIN
              and leaf.value not in get_local_scope(leaf)
              and leaf.value not in defined_names):
            inputs.append(leaf.value)

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
        return providers[variable]


class DefinitionsMapping:
    """
    Returns the available names (from import statements, function and class
    definitions) available for a given
    snippet. We use this to determine which names should not be considered
    inputs in tasks, since they are module names
    """
    def __init__(self, snippets):
        self._names = {
            name: set(find_defined_names(parso.parse(code)))
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
    # FIXME: this should also include defined classes and functions
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
