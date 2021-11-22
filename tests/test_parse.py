import pytest
import jupytext

from soorgeon import parse, exceptions

no_h2_headers = """# # Cell 0

1 + 1 # Cell 1

# # Cell 2
"""

all_h2 = """# ## Cell 0

1 + 1 # Cell 1

# ## Cell 2

2 + 2 # Cell 3

# ## Cell 4
"""

mixed = """# # Cell 0

1 + 1 # Cell 1

# ## Cell 2

2 + 2 # Cell 3

# ## Cell 4
"""

long_md = """# ## H2
# with more text here

1 + 1

# ## Another

2 + 2
"""

h1_next_to_h2 = """# # H1
# ## One

1 + 1

# ## Another

2 + 2
"""

exploratory = """# # Exploratory data analysis
#
# ## Load

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris(as_frame=True)['data']

# ## Clean

df = df[df['petal length (cm)'] > 2]

# ## Plot

sns.histplot(df['petal length (cm)'])
"""

# case with where cell only has H2 and H2 + more stuff
# edge case: H1, then H2 with no code in between, we should ignore that break


def _read(nb_str):
    return jupytext.reads(nb_str, fmt='py:light')


def test_find_breaks_error_if_no_h2_headers(tmp_empty):
    nb = _read(no_h2_headers)

    with pytest.raises(exceptions.InputError) as excinfo:
        parse.find_breaks(nb)

    assert 'Expected to have at least one' in str(excinfo.value)


@pytest.mark.parametrize('md, expected', [
    ['## Header', 'header'],
    ['# H1\n## H2', 'h2'],
    ['  ##   H2', 'h2'],
    ['something', None],
    ['## Something\nignore me', 'something'],
])
def test_get_h2_header(md, expected):
    assert parse._get_h2_header(md) == expected


@pytest.mark.parametrize('nb_str, expected', [
    [mixed, [2, 4]],
    [long_md, [0, 2]],
    [h1_next_to_h2, [0, 2]],
    [exploratory, [0, 3, 5]],
])
def test_find_breaks(tmp_empty, nb_str, expected):
    assert parse.find_breaks(_read(nb_str)) == expected


@pytest.mark.parametrize('cells, breaks, expected', [
    [[1, 2, 3, 4], [1], [[1, 2, 3, 4]]],
    [[1, 2, 3, 4], [1, 2], [[1, 2], [3, 4]]],
])
def test_split_with_breaks(cells, breaks, expected):
    assert parse.split_with_breaks(cells, breaks) == expected


@pytest.mark.parametrize('nb_str, expected', [
    [all_h2, ['cell-0', 'cell-2', 'cell-4']],
    [long_md, ['h2', 'another']],
    [exploratory, ['load', 'clean', 'plot']],
])
def test_names_with_breaks(tmp_empty, nb_str, expected):
    nb = _read(nb_str)

    # TODO: maybe create a parser class that saves the found breaks?
    # or maybe make it part of the proto dagspec
    breaks = parse.find_breaks(nb)

    assert parse.names_with_breaks(nb.cells, breaks) == expected


# TODO: do we need roundtrip conversion? we'l only use this for static analysis
# so i think we're fine
mixed_expected = '1 + 1 # Cell 1\n2 + 2 # Cell 3'


@pytest.mark.parametrize('code, expected', [
    [mixed, mixed_expected],
])
def test_prototask_str(code, expected):
    assert str(parse.ProtoTask('name', _read(code).cells)) == expected


@pytest.mark.parametrize('cells_idx, expected', [
    [(0, 3), 'from sklearn.datasets import load_iris'],
])
def test_prototask_add_imports_cell(cells_idx, expected):
    cells = jupytext.reads(exploratory,
                           fmt='py:light').cells[cells_idx[0]:cells_idx[1]]
    pt = parse.ProtoTask('task', cells)
    cell = pt._add_imports_cell(exploratory,
                                add_pathlib_and_pickle=False,
                                definitions=None)
    assert cell['source'] == expected


# FIXME: ensure _add_imports_cell removes comments


@pytest.mark.parametrize('name, expected', [
    ['task', 'task'],
    ['a task', 'a-task'],
    ['a ta/sk', 'a-ta-sk'],
])
def test_sanitize_name(name, expected):
    assert parse._sanitize_name(name) == expected