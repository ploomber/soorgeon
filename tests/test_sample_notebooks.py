from functools import partial
from glob import glob
from pathlib import Path

import pytest
from ploomber.spec import DAGSpec
from conftest import PATH_TO_TESTS

from soorgeon import export
from soorgeon._kaggle import download_from_dataset, download_from_competition

path_to_nbs_root = str(Path(PATH_TO_TESTS, '..', '_kaggle', '*'))
path_to_nbs = [
    path for path in glob(path_to_nbs_root)
    if Path(path).is_dir() and not Path(path).name.startswith('_')
]
ids = [Path(path).name for path in path_to_nbs]

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
