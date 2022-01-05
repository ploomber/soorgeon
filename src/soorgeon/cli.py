import click

from soorgeon import export


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--log', '-l', default=None)
@click.option('--product-prefix',
              '-p',
              default=None,
              help='Prefix for all products')
def refactor(path, log, product_prefix):
    """Refactor a monolithic notebook
    """
    export.from_path(path, log, product_prefix=product_prefix)
