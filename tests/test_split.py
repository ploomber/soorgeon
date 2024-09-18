import pytest

from soorgeon import split, exceptions

from testutils import exploratory, mixed, _read


no_markdown_but_plain_text = """Cell 0: 0

Cell 1: 1

Cell 2: 2
"""

no_markdown_but_json = """{
    "Cell 0": "",
    "Cell 1": "",
    "Cell 2": ""
}
"""

no_h1_and_h2_headers = """# ### Cell 0

1 + 1 # ### Cell 1

# ### Cell 2
"""

no_h2_but_h1_headers = """# # Cell 0

1 + 1 # # Cell 1

# # Cell 2
"""

all_h2 = """# ## Cell 0

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

only_one_h2 = """# ## Cell 0

1 + 1 # Cell 1

# # Cell 2

2 + 2 # Cell 3

# # Cell 4
"""

only_one_h2_diff = """# # Cell 0

1 + 1 # Cell 1

# ## Cell 2

2 + 2 # Cell 3

# # Cell 4
"""

no_h2_but_h1_headers_error = (
    "Only H1 headings are found. " "At this time, only H2 headings are supported"
)

no_h1_and_h2_headers_error = "Expected notebook to have at least one"

only_one_h2_header_warning = (
    "Warning: refactoring successful " "but only one H2 heading detected,"
)


# case with where cell only has H2 and H2 + more stuff
# edge case: H1, then H2 with no code in between, we should ignore that break
@pytest.mark.parametrize(
    "md, expected_msg",
    [
        [only_one_h2, only_one_h2_header_warning],
        [only_one_h2_diff, only_one_h2_header_warning],
    ],
)
def test_find_breaks_warnings(md, expected_msg, tmp_empty, capsys):
    nb = _read(md)
    split.find_breaks(nb)
    captured = capsys.readouterr()
    assert expected_msg in captured.out


@pytest.mark.parametrize(
    "md, expected_msg",
    [
        [no_h2_but_h1_headers, no_h2_but_h1_headers_error],
        [no_markdown_but_json, no_h1_and_h2_headers_error],
        [no_markdown_but_plain_text, no_h1_and_h2_headers_error],
        [no_h1_and_h2_headers, no_h1_and_h2_headers_error],
    ],
)
def test_find_breaks_errors(md, expected_msg, tmp_empty, capsys):
    nb = _read(md)

    with pytest.raises(exceptions.InputError) as excinfo:
        split.find_breaks(nb)

    assert expected_msg in str(excinfo.value)


@pytest.mark.parametrize(
    "md, expected",
    [
        ["## Header", "header"],
        ["# H1\n## H2", "h2"],
        ["  ##   H2", "h2"],
        ["  ###   H3", None],
        ["something", None],
        ["## Something\nignore me", "something"],
    ],
)
def test_get_h2_header(md, expected):
    assert split._get_h2_header(md) == expected


@pytest.mark.parametrize(
    "md, expected",
    [
        ["# Header", "header"],
        ["# H1\n## H2", "h1"],
        [" \t #   H1", "h1"],
        ["  ##   H2", None],
        ["something", None],
        ["# Something\nignore me", "something"],
    ],
)
def test_get_h1_header(md, expected):
    assert split._get_h1_header(md) == expected


@pytest.mark.parametrize(
    "nb_str, expected",
    [
        [mixed, [2, 4]],
        [long_md, [0, 2]],
        [h1_next_to_h2, [0, 2]],
        [exploratory, [0, 3, 5]],
    ],
)
def test_find_breaks(tmp_empty, nb_str, expected):
    assert split.find_breaks(_read(nb_str)) == expected


@pytest.mark.parametrize(
    "cells, breaks, expected",
    [
        [[1, 2, 3, 4], [1], [[1, 2, 3, 4]]],
        [[1, 2, 3, 4], [1, 2], [[1, 2], [3, 4]]],
    ],
)
def test_split_with_breaks(cells, breaks, expected):
    assert split.split_with_breaks(cells, breaks) == expected


@pytest.mark.parametrize(
    "nb_str, expected",
    [
        [all_h2, ["cell-0", "cell-2", "cell-4"]],
        [long_md, ["h2", "another"]],
        [exploratory, ["load", "clean", "plot"]],
    ],
)
def test_names_with_breaks(tmp_empty, nb_str, expected):
    nb = _read(nb_str)

    # TODO: maybe create a parser class that saves the found breaks?
    # or maybe make it part of the proto dagspec
    breaks = split.find_breaks(nb)

    assert split.names_with_breaks(nb.cells, breaks) == expected


# FIXME: ensure _add_imports_cell removes comments


@pytest.mark.parametrize(
    "name, expected",
    [
        ["task", "task"],
        ["a task", "a-task"],
        ["a ta/sk", "a-ta-sk"],
        ["some_task", "some-task"],
        ["this & that", "this-that"],
        ["some-task", "some-task"],
        ["`some_function()`", "-some-function-"],
        ["some.task", "some-task"],
        ["1.1 some stuff", "section-1-1-some-stuff"],
    ],
)
def test_sanitize_name(name, expected):
    assert split._sanitize_name(name) == expected
