from pathlib import Path

import yaml
import jupytext
import pytest
from click.testing import CliRunner

from soorgeon import cli
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
def test_refactor(tmp_empty, args, product_prefix):
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
    [['nb.py'], 'pkl', 'ploomber'],
    [['nb.py', '--df-format', 'parquet'], 'parquet', 'ploomber\npyarrow'],
    [['nb.py', '--df-format', 'csv'], 'csv', 'ploomber'],
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
    [imports_pyarrow, 'ploomber\npyarrow'],
    [imports_fastparquet, 'fastparquet\nploomber'],
    [imports_nothing, 'ploomber\npyarrow'],
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


def test_single_task(tmp_empty):
    jupytext.write(jupytext.reads(simple, fmt='py:light'), 'nb.ipynb')

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.ipynb', '--single-task'])

    assert result.exit_code == 0

    with Path('pipeline.yaml').open() as f:
        spec = yaml.safe_load(f)

    assert spec == {
        'tasks': [{
            'source': 'nb.py',
            'product': 'products/nb-report.ipynb',
            'static_analysis': False
        }]
    }


def test_suggest_single_task_if_failed_to_refactor(tmp_empty):
    Path('nb.py').write_text("""
# ## header

# this will break because of the missing : after the if keyword
if something
    pass
""")

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ['nb.py'])

    assert result.exit_code == 1
    assert 'soorgeon refactor nb.py --single-task' in result.output


# adds import if needed / and doesn't add import pickle
