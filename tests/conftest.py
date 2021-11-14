from pathlib import Path
import os

import jupytext
import pytest

from soorgeon.export import NotebookExporter


def path_to_tests():
    return Path(__file__).absolute().parent


def path_to_assets():
    return path_to_tests() / 'assets'


def read_nb(name):
    path = path_to_assets() / f'nb-{name}.py'
    return Path(path).read_text()


def read_snippets(name):
    ne = NotebookExporter(jupytext.reads(read_nb('ml'), fmt='py:percent'))
    return ne._snippets


@pytest.fixture
def tmp_empty(tmp_path):
    """
    Create temporary path using pytest native fixture,
    them move it, yield, and restore the original path
    """
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)
