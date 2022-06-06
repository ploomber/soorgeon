import click
from soorgeon import __version__, export
from soorgeon.clean import basic_clean


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
@click.option('--single-task',
              '-s',
              is_flag=True,
              help='Create a pipeline with a single task')
@click.option(
    '--file-format',
    '-f',
    default=None,
    type=click.Choice(('py', 'ipynb')),
    help=('Format for pipeline tasks, if empty keeps the same format '
          'as the input'))
def refactor(path, log, product_prefix, df_format, single_task, file_format):
    """
    Refactor a monolithic notebook.

    $ soorgeon refactor nb.ipynb

    * Sections must be separated by markdown H2 headings

    * Star imports (from math import *) not supported

    * Functions should not use global variables

    User guide: https://github.com/ploomber/soorgeon/blob/main/doc/guide.md
    """
    export.refactor(path,
                    log,
                    product_prefix=product_prefix,
                    df_format=df_format,
                    single_task=single_task,
                    file_format=file_format)

    click.secho(f'Finished refactoring {path!r}, use Ploomber to continue.',
                fg='green')

    click.echo("""
Install dependencies (this will install ploomber):
    $ pip install -r requirements.txt

List tasks:
    $ ploomber status

Execute pipeline:
    $ ploomber build

Plot pipeline:
    $ ploomber plot

* Documentation: https://docs.ploomber.io
* Jupyter integration: https://ploomber.io/s/jupyter
* Other editors: https://ploomber.io/s/editors
""")


@cli.command()
@click.argument("filename", type=click.Path(exists=True))
def clean(filename):
    """
    Clean a .py or .ipynb file (applies black and isort).

    $ soorgeon clean path/to/script.py
    or
    $ soorgeon clean path/to/notebook.ipynb

    """
    basic_clean(filename)
