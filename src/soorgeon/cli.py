import click

from soorgeon import export


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
def refactor(path):
    """Refactor a monolithic notebook
    """
    export.from_path(path)
