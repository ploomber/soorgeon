import os
from pathlib import Path


import pytest
from ploomber.spec import DAGSpec

from soorgeon import export
from soorgeon._pygithub import download_directory


dir_names = [
    "titanic-logistic-regression-with-python",
    "customer-segmentation-clustering",
    "intro-to-time-series-forecasting",
    "feature-selection-and-data-visualization",
    "linear-regression-house-price-prediction",
    "look-at-this-note-feature-engineering-is-easy",
]


@pytest.mark.parametrize("dir", dir_names, ids=list(dir_names))
def test_notebooks(tmp_empty, dir):
    download_directory(dir)
    path = os.getcwd()
    export.from_path(Path(path, "nb.py"), py=True)

    dag = DAGSpec("pipeline.yaml").to_dag()
    dag.build()
