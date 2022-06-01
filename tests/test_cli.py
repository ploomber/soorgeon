from unittest.mock import Mock
from pathlib import Path

import yaml
import jupytext
import pytest
from click.testing import CliRunner

from soorgeon import cli, export
from ploomber.spec import DAGSpec

simple = """# ## Cell 0

x = 1

# ## Cell 2

y = x + 1

# ## Cell 4

z = y + 1
"""


@pytest.mark.parametrize('args, product_prefix', [
    [['nb.py'], 'output'],
    [['nb.py', '--product-prefix', 'another'], 'another'],
    [['nb.py', '-p', 'another'], 'another'],
])
def test_refactor_product_prefix(tmp_empty, args, product_prefix):
    Path('nb.py').write_text(simple)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    spec = DAGSpec('pipeline.yaml')

    paths = [
        i for product in [t['product'].values() for t in spec['tasks']]
        for i in product
    ]

    assert result.exit_code == 0
    assert all([p.startswith(product_prefix) for p in paths])


@pytest.mark.parametrize('input_, out_ext, args', [
    ['nb.py', 'py', ['nb.py']],
    ['nb.ipynb', 'ipynb', ['nb.ipynb']],
    ['nb.py', 'ipynb', ['nb.py', '--file-format', 'ipynb']],
    ['nb.ipynb', 'py', ['nb.ipynb', '--file-format', 'py']],
])
def test_refactor_file_format(tmp_empty, input_, out_ext, args):
    jupytext.write(jupytext.reads(simple, fmt='py:light'), input_)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    assert result.exit_code == 0

    # test the output file has metadata, otherwise it may fail to execute
    # if missing the kernelspec info
    assert jupytext.read(Path('tasks', f'cell-0.{out_ext}')).metadata
    assert jupytext.read(Path('tasks', f'cell-2.{out_ext}')).metadata
    assert jupytext.read(Path('tasks', f'cell-4.{out_ext}')).metadata


with_dfs = """\
# ## first

df = 1

# ## second

df_2 = df + 1

"""

mixed = """\
# ## first

df = 1
x = 2

# ## second

df_2 = x + df + 1

"""


@pytest.mark.parametrize('args, ext, requirements', [
    [['nb.py'], 'pkl', 'ploomber>=0.14.7'],
    [['nb.py', '--df-format', 'parquet'], 'parquet',
     'ploomber>=0.14.7\npyarrow'],
    [['nb.py', '--df-format', 'csv'], 'csv', 'ploomber>=0.14.7'],
],
                         ids=[
                             'none',
                             'parquet',
                             'csv',
                         ])
@pytest.mark.parametrize('nb, products_expected', [
    [
        simple,
        [
            'output/cell-0-x.pkl',
            'output/cell-0.ipynb',
            'output/cell-2-y.pkl',
            'output/cell-2.ipynb',
            'output/cell-4.ipynb',
        ]
    ],
    [
        with_dfs,
        [
            'output/first-df.{ext}',
            'output/first.ipynb',
            'output/second.ipynb',
        ]
    ],
    [
        mixed,
        [
            'output/first-x.pkl',
            'output/first-df.{ext}',
            'output/first.ipynb',
            'output/second.ipynb',
        ]
    ],
],
                         ids=[
                             'simple',
                             'with-dfs',
                             'mixed',
                         ])
def test_refactor_df_format(tmp_empty, args, ext, nb, products_expected,
                            requirements):
    Path('nb.py').write_text(nb)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    spec = DAGSpec('pipeline.yaml')

    paths = [
        i for product in [t['product'].values() for t in spec['tasks']]
        for i in product
    ]

    assert result.exit_code == 0
    assert set(paths) == set(p.format(ext=ext) for p in products_expected)

    content = ('# Auto-generated file'
               f', may need manual editing\n{requirements}\n')
    assert Path('requirements.txt').read_text() == content


imports_pyarrow = """\
# ## first

import pyarrow

df = 1

# ## second

df_2 = df + 1

"""

imports_fastparquet = """\
# ## first

df = 1

# ## second

import fastparquet

df_2 = df + 1

"""

imports_nothing = """\
# ## first

df = 1

# ## second

df_2 = df + 1

"""


@pytest.mark.parametrize('nb, requirements', [
    [imports_pyarrow, 'ploomber>=0.14.7\npyarrow'],
    [imports_fastparquet, 'fastparquet\nploomber>=0.14.7'],
    [imports_nothing, 'ploomber>=0.14.7\npyarrow'],
],
                         ids=[
                             'pyarrow',
                             'fastparquet',
                             'nothing',
                         ])
def test_refactor_parquet_requirements(tmp_empty, nb, requirements):
    Path('nb.py').write_text(nb)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.py', '--df-format', 'parquet'])

    assert result.exit_code == 0
    content = ('# Auto-generated file'
               f', may need manual editing\n{requirements}\n')
    assert Path('requirements.txt').read_text() == content


@pytest.mark.parametrize('input_, backup, file_format, source', [
    ['nb.ipynb', 'nb-backup.ipynb', [], 'nb.ipynb'],
    ['nb.py', 'nb-backup.py', [], 'nb.py'],
    ['nb.ipynb', 'nb-backup.ipynb', ['--file-format', 'py'], 'nb.py'],
    ['nb.py', 'nb-backup.py', ['--file-format', 'ipynb'], 'nb.ipynb'],
])
def test_single_task(tmp_empty, input_, backup, file_format, source):
    jupytext.write(jupytext.reads(simple, fmt='py:light'), input_)

    runner = CliRunner()
    result = runner.invoke(cli.refactor,
                           [input_, '--single-task'] + file_format)

    assert result.exit_code == 0

    with Path('pipeline.yaml').open() as f:
        spec = yaml.safe_load(f)

    assert spec == {
        'tasks': [{
            'source': source,
            'product': 'products/nb-report.ipynb',
        }]
    }

    # test the output file has metadata, otherwise it may fail to execute
    # if missing the kernelspec info
    assert jupytext.read(Path(source)).metadata
    assert jupytext.read(Path(backup)).metadata


@pytest.mark.parametrize('code', [
    """
# ## header

if something
    pass
""", """
# ## header

y = x + 1
"""
],
                         ids=[
                             'syntax-error',
                             'undefined-name',
                         ])
def test_doesnt_suggest_single_task_if_nb_cannot_run(tmp_empty, code):
    Path('nb.py').write_text(code)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.py'])

    assert result.exit_code == 1
    assert 'soorgeon refactor nb.py --single-task' not in result.output


@pytest.mark.parametrize('code', [
    """
from math import *
""", """
y = 1

def x():
    return y
""", """
x = 1
"""
],
                         ids=[
                             'star-import',
                             'fn-with-global-vars',
                             'missing-h2-heading',
                         ])
def test_doesnt_suggest_single_task_if_nb_can_run(tmp_empty, code):
    Path('nb.py').write_text(code)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.py'])

    assert result.exit_code == 1
    assert 'soorgeon refactor nb.py --single-task' in result.output


def test_suggests_single_task_if_export_crashes(tmp_empty, monkeypatch):
    monkeypatch.setattr(export.NotebookExporter, 'export',
                        Mock(side_effect=KeyError))

    Path('nb.py').write_text(simple)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.py'])

    assert result.exit_code == 1
    assert 'soorgeon refactor nb.py --single-task' in result.output


# adds import if needed / and doesn't add import pickle


def test_clean_py():
    Path('nb.py').write_text(simple)

    runner = CliRunner()
    runner.invoke(cli.refactor, ['nb.py'])
    result = runner.invoke(cli.clean, ['tasks/cell-2.py'])
    assert result.exit_code == 0
    # black
    assert "1 file reformatted." in result.output
    # isort
    assert "Fixing" in result.output
    # end of basic_clean()
    assert "Finished cleaning tasks/cell-2.py" in result.output


def test_clean_ipynb():
    nb_ = jupytext.reads(simple, fmt='py:light')
    jupytext.write(nb_, 'nb.ipynb')

    runner = CliRunner()
    runner.invoke(cli.refactor, ['nb.ipynb'])
    result = runner.invoke(cli.clean, ['tasks/cell-2.ipynb'])

    assert result.exit_code == 0
    assert "Generating intermadiate py files" in result.output
    # black
    assert "1 file reformatted." in result.output
    # isort
    assert "Fixing" in result.output
    # end of basic_clean()
    assert "Finished cleaning tasks/cell-2.ipynb" in result.output


def test_clean_no_task():
    nb_ = jupytext.reads(simple, fmt='py:light')
    jupytext.write(nb_, 'nb.ipynb')

    runner = CliRunner()
    runner.invoke(cli.refactor, ['nb.ipynb'])
    result = runner.invoke(cli.clean, ['tasks/cell-9.ipynb'])

    assert result.exit_code == 2
    assert "Error: Invalid value for 'FILENAME'" in result.output
