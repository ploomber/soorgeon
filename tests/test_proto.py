import jupytext
import pytest

from testutils import exploratory, mixed, _read
from soorgeon import proto

# TODO: do we need roundtrip conversion? we'l only use this for static analysis
# so i think we're fine
mixed_expected = "1 + 1 # Cell 1\n2 + 2 # Cell 3"


@pytest.mark.parametrize(
    "code, expected",
    [
        [mixed, mixed_expected],
    ],
)
def test_prototask_str(code, expected):
    assert (
        str(
            proto.ProtoTask(
                "name", _read(code).cells, df_format=None, serializer=None, py=True
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "cells_idx, expected",
    [
        [(0, 3), "from sklearn.datasets import load_iris"],
    ],
)
def test_prototask_add_imports_cell(cells_idx, expected):
    cells = jupytext.reads(exploratory, fmt="py:light").cells[
        cells_idx[0] : cells_idx[1]
    ]
    pt = proto.ProtoTask("task", cells, df_format=None, serializer=None, py=True)
    cell = pt._add_imports_cell(
        exploratory,
        add_pathlib_and_pickle=False,
        definitions=None,
        df_format=None,
        serializer=None,
    )
    assert cell["source"] == expected
