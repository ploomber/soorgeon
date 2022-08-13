import click
import tempfile
import jupytext
import papermill as pm
from click.exceptions import ClickException
from papermill.exceptions import PapermillExecutionError
from os.path import abspath, dirname, splitext, join
from soorgeon.telemetry import telemetry
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
@click.argument("output_filename",
                type=click.Path(exists=False),
                required=False)
def test(filename, output_filename):
    """
    check if a .py or .ipynb file runs.

    $ soorgeon test path/to/script.py

    Optionally, set the path to the output notebook:

    $ soorgeon test path/to/notebook.ipynb path/to/output.ipynb
    or
    $ soorgeon test path/to/notebook.py path/to/output.ipynb

    """
    name, extension = splitext(filename)
    directory = dirname(abspath(filename))
    # save to {name}-soorgeon-test.ipynb by default
    if not output_filename:
        output_filename = join(directory, f"{name}-soorgeon-test.ipynb")
    else:
        output_filename = join(directory, output_filename)
    if extension.lower() == '.py':
        nb = jupytext.read(filename)
        # convert ipynb to py and create a temp file in current directory
        with tempfile.NamedTemporaryFile(suffix=".ipynb",
                                         delete=True,
                                         dir=directory) as temp_file:
            jupytext.write(nb, temp_file.name)
            _test(temp_file.name, output_filename)
    else:
        _test(filename, output_filename)


@telemetry.log_call('test')
def _test(filename, output_filename):
    CONTACT_MESSAGE = "An error happened when executing the notebook, " \
                      "contact us for help: https://ploomber.io/community"
    try:
        pm.execute_notebook(filename, output_filename, kernel_name='python3')
    except PapermillExecutionError as err:
        error_traceback = err.traceback
        error_suggestion_dict = {
            "ModuleNotFoundError":
            "Some packages are missing, please install them "
            "with 'pip install {package-name}'\n",
            "AttributeError":
            "AttributeErrors might be due to changes in the libraries "
            "you're using.\n",
            "SyntaxError":
            "There are syntax errors in the notebook.\n",
        }
        for error, suggestion in error_suggestion_dict.items():
            if any(error in error_line for error_line in error_traceback):
                click.secho(f"""\
{error} encountered while executing the notebook: {err}
{suggestion}
Output notebook: {output_filename}\n""",
                            fg='red')
                raise ClickException(CONTACT_MESSAGE)

        click.secho(f"""\
Error encountered while executing the notebook: {err}

Output notebook: {output_filename}\n""",
                    fg='red')
        raise ClickException(CONTACT_MESSAGE)
    except Exception as err:
        # handling errors other than PapermillExecutionError
        error_type = type(err).__name__
        click.echo(f"""\
{error_type} encountered while executing the notebook: {err}

Output notebook: {output_filename}""")
        raise ClickException(CONTACT_MESSAGE)
    else:
        click.secho(f"""\
Finished executing {filename}, no error encountered.
Output notebook: {output_filename}\n""",
                    fg='green')
