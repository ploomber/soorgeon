import click

from soorgeon import export


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--log', '-l', default=None)
def refactor(path, log):
    """Refactor a monolithic notebook
    """
    export.from_path(path, log)
