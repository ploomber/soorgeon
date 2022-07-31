import click
import tempfile
import jupytext
from os.path import abspath, dirname
from soorgeon import __version__, export
from soorgeon import clean as clean_module


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
@click.option('--serializer',
              '-z',
              default=None,
              type=click.Choice(('cloudpickle', 'dill')),
              help='Serializer for non-picklable data')
def refactor(path, log, product_prefix, df_format, single_task, file_format,
             serializer):
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
                    file_format=file_format,
                    serializer=serializer)

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
    clean_module.basic_clean(filename)


@cli.command()
@click.argument("filename", type=click.Path(exists=True))
def lint(filename):
    """
    Lint a .py or .ipynb file using flake8

    $ soorgeon lint path/to/script.py

    or

    $ soorgeon lint path/to/notebook.ipynb
    """
    clean_module.lint(filename)


@cli.command()
@click.argument("filename", type=click.Path(exists=True))
def test(filename):
    """
    check if a .py or .ipynb file runs.

    $ soorgeon test path/to/script.py

    or

    $ soorgeon test path/to/notebook.ipynb
    """
    if filename.lower().endswith(".ipynb"):
        nb = jupytext.read(filename)
        # convert ipynb to py and create a temp file in current directory
        directory = dirname(abspath(filename))
        with tempfile.NamedTemporaryFile(suffix=".py",
                                         delete=True,
                                         dir=directory) as temp_file:
            jupytext.write(nb, temp_file.name)
            _test(temp_file.name)
    else:
        _test(filename)


def _test(filename):
    try:
        exec(open(filename).read())
        click.echo(f"Finished executing {filename}, no error encountered")
    except (ModuleNotFoundError, AttributeError, SyntaxError) as err:
        error_suggestion_dict = {
            "ModuleNotFoundError": "create a virtualenv, and"
            " adding a requirements.txt with the package",
            "AttributeError": "downgrade some libraries",
            "SyntaxError": "check syntax",
        }
        error_type = type(err).__name__
        click.echo(f"""
        {error_type} encountered while executing the notebook: {err}

        It is recommended to {error_suggestion_dict[error_type]}
        """)
    except Exception as err:
        error_type = type(err).__name__
        click.echo(f"""
        {error_type} encountered while executing the notebook: {err}

        Checkout how to debug notebooks:
        https://docs.ploomber.io/en/latest/user-guide/debugging.html
        """)
