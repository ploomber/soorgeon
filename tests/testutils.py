import parso


def get_first_leaf_with_value(code, value):
    leaf = parso.parse(code).get_first_leaf()

    while leaf:
        if leaf.value == value:
            return leaf

        leaf = leaf.get_next_leaf()

    raise ValueError(f'could not find leaf with value {value}')
