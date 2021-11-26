import parso
import pytest

from soorgeon import definitions

simple_imports = """
import pandas as pd
import numpy as np
"""


@pytest.mark.parametrize('code, expected', [
    [
        simple_imports, {
            'np': 'import numpy as np',
            'pd': '\nimport pandas as pd'
        }
    ],
])
def test_find_defined_names_from_imports(code, expected):
    assert definitions.from_imports(parso.parse(code)) == expected


@pytest.mark.parametrize(
    'code, expected',
    [["""
def x():
    pass
""", {
        'x': '\ndef x():\n    pass'
    }], ["""
class X:
    pass
""", {
        'X': '\nclass X:\n    pass'
    }],
     [
         """
def x():
    pass

class X:
    pass
""", {
             'X': '\nclass X:\n    pass',
             'x': '\ndef x():\n    pass'
         }
     ]])
def test_find_defined_names_from_def_and_class(code, expected):
    out = (definitions.from_def_and_class(parso.parse(code)))
    assert out == expected
