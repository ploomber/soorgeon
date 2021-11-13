import parso


def find_inputs_and_outputs(code_str):
    """
    Given a Python code string, find which variables the code consumes (not
    declared in the snipped) and which ones it exposes (declared in the
    snippet)
    """
    tree = parso.parse(code_str)
    leaf = tree.get_first_leaf()

    inputs, outputs = [], set()

    while leaf:
        if leaf.type == 'operator' and leaf.value == '=':
            next = leaf.get_next_sibling()
            previous = leaf.get_previous_leaf()

            try:
                children = next.children
            except AttributeError:
                inputs_current = []
            else:
                inputs_current = [
                    e.value for e in children if e.type == 'name'
                ]

            for variable in inputs_current:
                # only mark a variable as input if it hasn't been defined
                # locally
                if variable not in outputs:
                    inputs.append(variable)

            outputs.add(previous.value)

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

    io = {
        snippet_name: find_inputs_and_outputs(snippet)
        for snippet_name, snippet in snippets.items()
    }

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
