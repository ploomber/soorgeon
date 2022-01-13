import click

from soorgeon import __version__, export


@click.group()
@click.version_option(__version__)
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
    click.secho(f'Finished refactoring {path!r}, use Ploomber to continue.',
                fg='green')
    click.echo("""
Install:
    $ pip install ploomber

List tasks:
    $ ploomber status

Execute pipeline:
    $ ploomber build

Plot pipeline (note: this requires pygraphviz):
    $ ploomber plot

Documentation: https://docs.ploomber.io
""")
