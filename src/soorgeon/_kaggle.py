"""
CLI for downloading Kaggle notebooks for integration testing
"""
from functools import partial
import zipfile
import shutil
from pathlib import PurePosixPath, Path

import click
import jupytext
import papermill as pm
from kaggle import api


def process_index(index):
    return {PurePosixPath(i['url']).name: _add_partial(i) for i in index}


def _add_partial(d):
    url_data = PurePosixPath(d['data'])
    is_competition = 'c' in url_data.parts
    fn = download_from_competition if is_competition else download_from_dataset

    if is_competition:
        name = url_data.name
    else:
        name = str(PurePosixPath(*url_data.parts[-2:]))

    kwargs = dict(name=name)

    if d.get('files'):
        kwargs['files'] = d['files']

    return {**d, **dict(partial=partial(fn, **kwargs))}


def download_from_competition(name, files=None):
    # FIXME: add support for more than one file
    api.competition_download_cli(name, file_name=files)

    if not files:
        with zipfile.ZipFile(f'{name}.zip', 'r') as file:
            file.extractall('input')
    else:
        Path('input').mkdir()
        shutil.move(files, Path('input', files))


def download_from_dataset(name):
    api.dataset_download_cli(name, unzip=True, path='input')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('kernel_path')
def notebook(kernel_path):
    click.echo('Downloading notebook...')
    name = PurePosixPath(kernel_path).name
    api.kernels_pull_cli(kernel=kernel_path, path=name)

    click.echo('Converting to .py...')
    ipynb = Path(name, f'{name}.ipynb')
    py = Path(name, 'nb.py')
    nb = jupytext.read(ipynb)
    # TODO: remove cells that are !pip install ...
    jupytext.write(nb, py, fmt='py:percent')
    ipynb.unlink()


# FIXME: have a single command that detects if it's a competition or not
# update CONTRIBUTING.md
# FIXME: add files arg
@cli.command()
@click.argument('name')
def competition(name):
    download_from_competition(name=name)


@cli.command()
@click.argument('name')
def dataset(name):
    download_from_dataset(name=name)


@cli.command()
@click.argument('path', type=click.Path(exists=True))
def test(path):
    nb = jupytext.read(path, fmt='py:percent')
    click.echo('Generating test.ipynb...')
    jupytext.write(nb, 'test.ipynb')
    click.echo('Executing test.ipynb...')
    pm.execute_notebook('test.ipynb', 'test.ipynb')


if __name__ == '__main__':
    cli()
