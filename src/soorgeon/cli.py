import click

from soorgeon import __version__, export


@click.group()
@click.version_option(__version__)
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--log', '-l', default=None)
@click.option(
    '--df-format',
    '-d',
    default=None,
    type=click.Choice(('parquet', 'csv')),
    help='Format for variables with the df prefix. Otherwise uses pickle')
@click.option('--product-prefix',
              '-p',
              default=None,
              help='Prefix for all products')
def refactor(path, log, product_prefix, df_format):
    """
    Refactor a monolithic notebook.

    * Sections must be separated by markdown H2 headings

    * Star imports (from math import *) not supported

    * Functions should not use global variables

    $ soorgeon refactor nb.ipynb

    User guide: https://github.com/ploomber/soorgeon/blob/main/doc/guide.md
    """
    export.from_path(path,
                     log,
                     product_prefix=product_prefix,
                     df_format=df_format)
    click.secho(f'Finished refactoring {path!r}, use Ploomber to continue.',
                fg='green')

    click.echo("""
Install dependencies (this will install ploomber):
    $ pip install -r requirements.txt

List tasks:
    $ ploomber status

Execute pipeline:
    $ ploomber build

Plot pipeline (this requires pygraphviz, which isn't installed by default):
    $ ploomber plot

Documentation: https://docs.ploomber.io
""")
