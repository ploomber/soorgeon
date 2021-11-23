import shutil
from glob import glob
from pathlib import Path

import pytest
from ploomber.spec import DAGSpec
from conftest import PATH_TO_TESTS

from soorgeon import export

path_to_nbs_root = str(Path(PATH_TO_TESTS, '..', '_kaggle', '*'))
path_to_nbs = [path for path in glob(path_to_nbs_root) if Path(path).is_dir()]
ids = [Path(path).name for path in path_to_nbs]


@pytest.mark.parametrize('path', path_to_nbs, ids=ids)
def test_notebooks(tmp_empty, path):
    shutil.copytree(Path(path, 'input'), 'input')
    export.from_path(Path(path, 'nb.py'))

    dag = DAGSpec('pipeline.yaml').to_dag()
    dag.build()