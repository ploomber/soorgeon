import click
from os.path import exists
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
@click.argument("task-name")
@click.option("--deep", "-d", default=None)
def clean(task_name, deep):
    """
    Clean a refactored notebook task.

    $ soorgeon clean model-training

    """
    task_dir = "tasks"
    task_files = [
        f"{task_dir}/{task_name}.py", f"{task_dir}/{task_name}.ipynb"
    ]
    if not exists(f"{task_dir}"):
        click.echo("tasks directory not found, please refactor first!",
                   err=True)
    elif not any(exists(task_file) for task_file in task_files):
        click.echo(f"task {task_name} not found!", err=True)
    else:
        for taskfile in [f for f in task_files if exists(f)]:
            basic_clean(taskfile)
            if deep:
                pass  # TODO issue #49
