from functools import partial
import zipfile
import shutil
from glob import glob
from pathlib import Path

import pytest
from kaggle import api
from ploomber.spec import DAGSpec
from conftest import PATH_TO_TESTS

from soorgeon import export

path_to_nbs_root = str(Path(PATH_TO_TESTS, '..', '_kaggle', '*'))
path_to_nbs = [
    path for path in glob(path_to_nbs_root)
    if Path(path).is_dir() and not Path(path).name.startswith('_')
]
ids = [Path(path).name for path in path_to_nbs]


def download_from_competition(name, filename=None):
    api.competition_download_cli(name, file_name=filename)

    if not filename:
        with zipfile.ZipFile(f'{name}.zip', 'r') as file:
            file.extractall('input')
    else:
        Path('input').mkdir()
        shutil.move(filename, Path('input', filename))


def download_from_dataset(name):
    api.dataset_download_cli(name, unzip=True, path='input')


download = {
    'titanic-logistic-regression-with-python':
    partial(download_from_competition, name='titanic'),
    'customer-segmentation-clustering':
    partial(download_from_dataset,
            'imakash3011/customer-personality-analysis'),
    'intro-to-time-series-forecasting':
    partial(download_from_competition,
            name='acea-water-prediction',
            filename='Aquifer_Petrignano.csv'),
    'feature-selection-and-data-visualization':
    partial(download_from_dataset, name='uciml/breast-cancer-wisconsin-data'),
    'linear-regression-house-price-prediction':
    partial(download_from_dataset, name='vedavyasv/usa-housing'),
}


@pytest.mark.parametrize('path', path_to_nbs, ids=ids)
def test_notebooks(tmp_empty, path):
    name = Path(path).name
    download[name]()

    export.from_path(Path(path, 'nb.py'))

    dag = DAGSpec('pipeline.yaml').to_dag()
    dag.build()
