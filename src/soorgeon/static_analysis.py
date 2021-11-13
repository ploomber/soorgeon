from functools import reduce

import parso


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

    parent = leaf.parent

    try:
        left = parent.get_previous_sibling().value == '('
    except AttributeError:
        left = False

    try:
        right = parent.get_next_sibling().value == ')'
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

    defined_names_from_imports = find_defined_names_from_imports(tree)

    inputs, outputs = [], set()

    while leaf:
        if leaf.type == 'operator' and leaf.value == '=':
            next_s = leaf.get_next_sibling()
            previous = leaf.get_previous_leaf()

            try:
                children = next_s.children
            except AttributeError:
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
            # e.g. something(key=value)
            if previous.parent.type != 'argument':
                outputs.add(previous.value)

        # variables inside function calls are inputs
        # e.g., some_function(df)
        elif leaf.type == 'name' and inside_function_call(leaf):
            inputs.append(leaf.value)

        leaf = leaf.get_next_leaf()

    return set(inputs), outputs


def _map_outputs(name, outputs):
    return [(out, name) for out in outputs]


def _get_upstream(inputs, providers):
    return [providers[input_] for input_ in inputs]


def find_upstream(snippets):
    """
    Parameters
    ----------
    snippets : dict
        {snippet_name: snippet, ...}
    """
    # NOTE: to fix the load, clean plot test case, we should take into account
    # the order of snippets, and assume that earlier keys from from earlier nb
    # sections

    io = {
        snippet_name: find_inputs_and_outputs(snippet)
        for snippet_name, snippet in snippets.items()
    }

    # NOTE: I think we need to organize providers by key and order, so if
    # two tasks define the same variable, they receive a lower order if
    # the code appears in earlier cells. this way, when looking up providers,
    # we'll match the closest one (from the previous sections)
    providers = find_providers(io)

    upstream = {
        snippet_name: _get_upstream(v[0], providers)
        for snippet_name, v in io.items()
    }

    return upstream


def find_providers(io):
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
