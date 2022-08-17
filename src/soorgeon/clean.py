import shutil
import subprocess
import tempfile
from black import format_file_in_place, FileMode, WriteBack
from contextlib import contextmanager
from pathlib import Path

import click
import jupytext

from soorgeon.exceptions import BaseException
from soorgeon.telemetry import telemetry


def _jupytext_fmt(text, extension):
    """
    Determine the jupytext fmt string to use based on the content and extension
    """
    if extension != 'ipynb':
        fmt, _ = jupytext.guess_format(text, extension)
        fmt_final = f'{extension}:{fmt}'
    else:
        fmt_final = '.ipynb'

    return fmt_final


@telemetry.log_call('lint')
def lint(task_file):
    with get_file(task_file, write=False, output_ext=".py") as path:
        run_program(path, program='flake8', filename=task_file)


@telemetry.log_call('clean')
def basic_clean(task_file, program="black"):
    """
    Run basic clean (directly called by cli.clean())
    Generate intermediate files for ipynb
    """
    with get_file(task_file, write=True) as path:
        clean_py(path, task_file)

    click.echo(f"Finished cleaning {task_file}")


def clean_py(task_file_py, filename):
    # reformat with black
    black_result = format_file_in_place(task_file_py,
                                        fast=True,
                                        mode=FileMode(),
                                        write_back=WriteBack(1))
    if black_result:
        click.echo(f"Reformatted {filename} with black.")


def run_program(task_file_py, program, filename):
    """
    Run basic clean for py files
    (util method only called by basic_clean())
    """
    if shutil.which(program) is None:
        raise BaseException(f'{program} is missing, please install it with:\n'
                            f'pip install {program}\nand try again')
    # black
    result = subprocess.run([program, task_file_py],
                            text=True,
                            capture_output=True)

    click.echo(result.stdout.replace(str(task_file_py), filename))
    click.echo(result.stderr)


@contextmanager
def get_file(task_file, write=False, output_ext=".ipynb"):
    task_file = Path(task_file)
    # only works for black
    create_temp = task_file.suffix != output_ext
    text = task_file.read_text()

    if create_temp:
        nb = jupytext.reads(text)
        temp_path = tempfile.NamedTemporaryFile(suffix=output_ext,
                                                delete=False).name
        jupytext.write(nb, temp_path)
        path = Path(temp_path)

    else:
        path = task_file

    try:
        yield path
    finally:
        if write:
            jupytext.write(jupytext.read(path),
                           task_file,
                           fmt=_jupytext_fmt(text, task_file.suffix[1:]))

        if create_temp:
            Path(path).unlink()
