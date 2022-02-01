from pathlib import Path

import yaml
import pytest
from ploomber.spec import DAGSpec
from conftest import PATH_TO_TESTS

from soorgeon import export
from soorgeon._kaggle import process_index

_kaggle = Path(PATH_TO_TESTS, '..', '_kaggle')
path_to_index = _kaggle / 'index.yaml'
index_raw = yaml.safe_load(path_to_index.read_text())

index = process_index(index_raw)
path_to_nbs = [_kaggle / name for name in index]


@pytest.mark.parametrize('path', path_to_nbs, ids=list(index))
def test_notebooks(tmp_empty, path):
    name = Path(path).name
    download_fn = index[name]['partial']
    download_fn()

    export.from_path(Path(path, 'nb.py'), py=True)

    dag = DAGSpec('pipeline.yaml').to_dag()
    dag.build()
