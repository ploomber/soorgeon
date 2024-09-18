"""
Get some notebooks from kaggle, refactor and render DAG
"""

from glob import glob
from pathlib import Path

import pytest
from ploomber.spec import DAGSpec
from conftest import PATH_TO_TESTS

from soorgeon import export

_kaggle = Path(PATH_TO_TESTS, "..", "_kaggle", "_render")
path_to_nbs = glob(str(Path(_kaggle, "*", "*.py")))


def get_name(path):
    return Path(path).parent.name


names = [get_name(nb) for nb in path_to_nbs]


@pytest.mark.parametrize("path", path_to_nbs, ids=names)
def test_notebooks(tmp_empty, path):
    export.from_path(path, py=True)
    DAGSpec("pipeline.yaml").to_dag().render()
