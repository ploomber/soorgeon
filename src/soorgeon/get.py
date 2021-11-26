def first_expr_stmt_parent(node):
    parent = node.parent

    if not parent:
        return None

    while parent.type != 'expr_stmt':
        parent = parent.parent

        if not parent:
            break

    return parent


def first_non_atom_expr_parent(node):
    parent = node.parent

    # e.g., [x.attribute for x in range(10)]
    # x.attribute is an atom_expr
    while parent.type == 'atom_expr':
        parent = parent.parent

    return parent
