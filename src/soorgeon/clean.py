import subprocess
import isort
import click
import tempfile
import jupytext


def basic_clean(task_file):
    """Run basic clean, directly called by cli.clean
    Generate intermediate files for ipynb
    """
    if task_file.lower().endswith(".ipynb"):
        click.echo("Generating intermadiate py files for notebook cleaning")
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
    """Run basic clean for py files (core part)"""
    # commands to execute, which follow <exectutable> <filename> format
    commands = ["black"]
    for command in commands:
        result = subprocess.run([command, task_file_py],
                                text=True,
                                capture_output=True)
        click.echo(result)

    isort.file(task_file_py)
