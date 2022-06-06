import subprocess
import isort
import click
import tempfile
import jupytext
import shutil
from soorgeon.exceptions import BaseException


def basic_clean(task_file):
    """
    Run basic clean (directly called by cli.clean())
    Generate intermediate files for ipynb
    """
    if task_file.lower().endswith(".ipynb"):
        nb = jupytext.read(task_file)
        temp_path = tempfile.NamedTemporaryFile(suffix=".py",
                                                delete=False).name
        jupytext.write(nb, temp_path)
        basic_clean_py(temp_path)
        jupytext.write(jupytext.read(temp_path), task_file)
    else:
        basic_clean_py(task_file)
    click.echo(f"Finished cleaning {task_file}")


def basic_clean_py(task_file_py):
    """
    Run basic clean for py files
    (util method only called by basic_clean())
    """
    if shutil.which('black') is None:
        raise BaseException('black is missing, please install it with:\n'
                            'pip install black\nand try again')
    # black
    result = subprocess.run(["black", task_file_py],
                            text=True,
                            capture_output=True)
    click.echo(result.stderr)
    # isort
    isort.file(task_file_py)
