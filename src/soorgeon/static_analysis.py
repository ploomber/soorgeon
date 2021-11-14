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


def inside_function_call(leaf):
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


def find_inputs_and_outputs(code_str):
    """
    Given a Python code string, find which variables the code consumes (not
    declared in the snipped) and which ones it exposes (declared in the
    snippet)
    """
    tree = parso.parse(code_str)
    leaf = tree.get_first_leaf()

    # NOTE: we use this in find_inputs_and_outputs and ImportParser, maybe
    # move the functionality to a class so we only compute it once
    defined_names_from_imports = find_defined_names_from_imports(tree)

    inputs, outputs = [], set()

    while leaf:
        if leaf.type == 'operator' and leaf.value == '=':
            next_s = leaf.get_next_sibling()
            previous = leaf.get_previous_leaf()

            try:
                children = next_s.children
            except AttributeError:
                # could be keyword arguments inside a function call
                if leaf.parent.type == 'argument' and leaf.get_next_leaf(
                ).value not in _BUILTIN and leaf.get_next_leaf(
                ).type == 'name':
                    # TODO: there could be more than one
                    inputs_current = [leaf.get_next_leaf().value]
                else:
                    inputs_current = []
            else:
                if leaf.get_next_leaf().value in _BUILTIN:
                    inputs_current = []
                else:
                    inputs_current = [
                        e.value for e in children if e.type == 'name'
                        and e.value not in defined_names_from_imports
                    ]

            for variable in inputs_current:
                # only mark a variable as input if it hasn't been defined
                # locally
                if variable not in outputs:
                    inputs.append(variable)

            # ignore keyword arguments, they aren't outputs
            # e.g. 'key' in something(key=value)
            # also ignore previous if modifying an existing object
            # e.g.,
            # a = {}
            # a['x'] = 1
            # a.b = 1
            if (previous.parent.type != 'argument'
                    and not _modifies_existing_object(
                        leaf, outputs, defined_names_from_imports)):

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

        # variables inside function calls are inputs
        # e.g., some_function(df)
        # but ignore them if they have been locally defined
        elif (leaf.type == 'name' and inside_function_call(leaf)
              and leaf.value not in outputs):
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
    # FIXME: find_upstream already calls this, we should only compute it once
    return {
        snippet_name: find_inputs_and_outputs(snippet)
        for snippet_name, snippet in snippets.items()
    }


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
