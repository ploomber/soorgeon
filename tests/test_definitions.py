import parso
import pytest

from soorgeon import definitions

simple_imports = """
import pandas as pd
import numpy as np
"""

mixed_imports = """
import pandas
import numpy as np
from sklearn import ensemble
from another.sub import stuff
import matplotlib.pyplot as plt


import math
import ast as ast_
from random import choice
from collections.abc import Generator
from collections.abc import Generator as Gen
"""

relative_imports = """
from . import x
from .x import y
from ..x import y
"""

duplicated_imports = """
from sklearn import ensemble
from sklearn import linear_model
"""

comma_imports = """
from sklearn import ensemble, linear_model

from collections.abc import Generator, Collection
"""

two_import_as = """
import matplotlib.pyplot as plt, numpy as np
"""

two_imports = """
import numpy, pandas
"""


@pytest.mark.parametrize(
    "code, expected",
    [
        [simple_imports, {"np": "import numpy as np", "pd": "\nimport pandas as pd"}],
    ],
)
def test_from_imports(code, expected):
    assert definitions.from_imports(parso.parse(code)) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        [
            simple_imports,
            [
                "numpy",
                "pandas",
            ],
        ],
        [
            mixed_imports,
            [
                "another",
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
            ],
        ],
        [
            relative_imports,
            [],
        ],
        [
            duplicated_imports,
            ["scikit-learn"],
        ],
        [
            comma_imports,
            ["scikit-learn"],
        ],
        [
            two_import_as,
            [
                "matplotlib",
                "numpy",
            ],
        ],
        [
            two_imports,
            [
                "numpy",
                "pandas",
            ],
        ],
    ],
    ids=[
        "simple",
        "mixed",
        "relative",
        "duplicated",
        "comma",
        "two-import-as",
        "two-imports",
    ],
)
def test_packages_used(code, expected):
    assert definitions.packages_used(parso.parse(code)) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        [
            """
def x():
    pass
""",
            {"x": "\ndef x():\n    pass"},
        ],
        [
            """
class X:
    pass
""",
            {"X": "\nclass X:\n    pass"},
        ],
        [
            """
def x():
    pass

class X:
    pass
""",
            {"X": "\nclass X:\n    pass", "x": "\ndef x():\n    pass"},
        ],
    ],
)
def test_find_defined_names_from_def_and_class(code, expected):
    out = definitions.from_def_and_class(parso.parse(code))
    assert out == expected
