import zipfile
import shutil
from pathlib import PurePosixPath, Path

import click
import jupytext
from kaggle import api


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


@click.group()
def cli():
    pass


@cli.command()
@click.argument('kernel_path')
def notebook(kernel_path):
    click.echo('Downloading notebook...')
    name = PurePosixPath(kernel_path).name
    api.kernel_paths_pull_cli(kernel=kernel_path, path=name)

    click.echo('Converting to .py...')
    ipynb = Path(name, f'{name}.ipynb')
    py = Path(name, 'nb.py')
    nb = jupytext.read(ipynb)
    jupytext.write(nb, py, fmt='py:percent')
    ipynb.unlink()


@cli.command()
@click.argument('name')
def competition(name):
    download_from_competition(name=name)


@cli.command()
@click.argument('name')
def dataset(name):
    download_from_dataset(name=name)


if __name__ == '__main__':
    cli()