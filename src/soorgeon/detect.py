"""
Detect which kind of structure we're dealing with
"""
from soorgeon import get


def is_f_string(leaf):
    return leaf.type == 'fstring_start'


def is_funcdef(leaf):
    """
    Returns true if the leaf is the beginning of a function definition (def
    keyword)
    """
    return leaf.type == 'keyword' and leaf.value == 'def'


def is_lambda(leaf, raise_=False):
    """
    Returns true if the leaf is the beginning of a lambda definition
    """
    return leaf.type == 'keyword' and leaf.value == 'lambda'


def is_classdef(leaf):
    """
    Returns true if the leaf is the beginning of a class definition
    """
    return leaf.type == 'keyword' and leaf.value == 'class'


def is_for_loop(leaf):
    """
    Returns true if the leaf is the beginning of a for loop
    """
    has_suite_parent = False
    parent = leaf.parent

    while parent:
        if parent.type == 'suite':
            has_suite_parent = True

        if parent.type == 'for_stmt':
            return not has_suite_parent

        parent = parent.parent

    return False


def is_comprehension(leaf):
    """
    Return true if the leaf is the beginning of a list/set/dict comprehension.
    Returns true for generators as well
    """
    if leaf.type != 'operator' or leaf.value not in {'[', '(', '{'}:
        return False

    sibling = leaf.get_next_sibling()

    return (sibling.type in {'testlist_comp', 'dictorsetmaker'}
            and sibling.children[-1].type == 'sync_comp_for')


def is_context_manager(leaf):
    """
    Returns true if the leaf is the beginning of a context manager
    """
    has_suite_parent = False
    parent = leaf.parent

    while parent:
        if parent.type == 'suite':
            has_suite_parent = True

        if parent.type == 'with_stmt':
            return not has_suite_parent

        parent = parent.parent

    return False


def is_left_side_of_assignment(node):
    to_check = get.first_expr_stmt_parent(node)

    if not to_check:
        return False

    return to_check.children[1].value == '='


# FIXME: delete
def is_inside_list_comprehension(node):
    parent = get.first_non_atom_expr_parent(node)

    return (parent.type == 'testlist_comp'
            and parent.children[1].type == 'sync_comp_for')


def is_inside_funcdef(leaf):
    parent = leaf.parent

    while parent:
        if parent.type == 'funcdef':
            return True

        parent = parent.parent

    return False


def is_inside_function_call(leaf):
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

    # check if the node is inside parenhesis: function(df)
    # or a parent of the node: function(df.something)
    # NOTE: do we need more checks to ensure we're in the second case?
    # maybe check if we have an actual dot, or we're using something like
    # df[key]?

    node = leaf

    while node:
        inside = is_inside_parenthesis(node)

        if inside:
            return True
        else:
            node = node.parent

    return False


def is_inside_parenthesis(node):
    try:
        left = node.get_previous_sibling().value == '('
    except AttributeError:
        left = False

    try:
        right = node.get_next_sibling().value == ')'
    except AttributeError:
        right = False

    try:
        # to prevent (1, 2, 3) being detected as a function call
        has_name = node.get_previous_sibling().get_previous_leaf(
        ).type == 'name'
    except AttributeError:
        has_name = False

    return left and right and has_name


def is_accessing_variable(leaf):
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
