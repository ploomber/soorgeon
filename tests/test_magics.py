import pytest
import jupytext

from soorgeon import magics

source = """\
# ## first

# + language="bash"
# ls

# + language="html"
# <br>hi
# -

# ## second

# %timeit 1 + 1

# %cd x

# %%capture
print('x')

# +
! echo hello
"""


@pytest.mark.parametrize(
    "source, expected",
    [
        ["%%html\na\nb", "# [magic] %%html\n# [magic] a\n# [magic] b"],
        ["%%capture\na\nb", "# [magic] %%capture\na\nb"],
        ["%%timeit\na\nb", "# [magic] %%timeit\na\nb"],
        ["%%time\na\nb", "# [magic] %%time\na\nb"],
        ["%time 1\n2\n%time 3", "# [magic] %time 1\n2\n# [magic] %time 3"],
    ],
    ids=[
        "another-language",
        "inline-python",
        "inline-python-2",
        "inline-python-3",
        "line-magics",
    ],
)
def test_comment_if_ipython_magic(source, expected):
    assert magics._comment_if_ipython_magic(source) == expected


def test_comment_magics():
    nb = jupytext.reads(source, fmt="py:light")

    nb_new = magics.comment_magics(nb)

    assert [c["source"] for c in nb_new.cells] == [
        "## first",
        "# [magic] %%bash\n# [magic] ls",
        "# [magic] %%html\n# [magic] <br>hi",
        "## second",
        "# [magic] %timeit 1 + 1",
        "# [magic] %cd x",
        "# [magic] %%capture\nprint('x')",
        "# [magic] ! echo hello",
    ]


@pytest.mark.parametrize(
    "line, expected",
    [
        ["# [magic] %%timeit something()", "%%timeit something()"],
        ["# [magic] %timeit something()", "%timeit something()"],
        ["# [magic] %some_magic another()", "%some_magic another()"],
    ],
)
def test_uncomment_magic(line, expected):
    assert magics._uncomment_magic(line) == expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ["# [magic] %%timeit something()", False],
        ["something() # [magic] %timeit", ("something()", "%timeit")],
        ["another() # [magic] %time", ("another()", "%time")],
    ],
)
def test_is_commented_line_magic(line, expected):
    assert magics._is_commented_line_magic(line) == expected


def test_uncomment_magics_cell():
    nb = jupytext.reads(source, fmt="py:light")

    nb_new = magics.comment_magics(nb)

    assert [magics._uncomment_magics_cell(c["source"]) for c in nb_new.cells] == [
        "## first",
        "%%bash\nls",
        "%%html\n<br>hi",
        "## second",
        "%timeit 1 + 1",
        "%cd x",
        "%%capture\nprint('x')",
        "! echo hello",
    ]


def test_uncomment_magics():
    nb = jupytext.reads(source, fmt="py:light")

    nb_new = magics.comment_magics(nb)
    nb_out = magics.uncomment_magics(nb_new)

    assert [c["source"] for c in nb_out.cells] == [
        "## first",
        "%%bash\nls",
        "%%html\n<br>hi",
        "## second",
        "%timeit 1 + 1",
        "%cd x",
        "%%capture\nprint('x')",
        "! echo hello",
    ]


@pytest.mark.parametrize(
    "line, expected",
    [
        ["%timeit x = 1", "x = 1 # [magic] %timeit"],
        ["%time x = 1", "x = 1 # [magic] %time"],
        ["     %time x = 1", "x = 1 # [magic] %time"],
    ],
    ids=[
        "time",
        "timeit",
        "leading-whitespace",
    ],
)
def test_comment_ipython_line_magic(line, expected):
    magic = magics._is_ipython_line_magic(line)
    assert magics._comment_ipython_line_magic(line, magic) == expected
