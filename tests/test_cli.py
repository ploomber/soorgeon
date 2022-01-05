from pathlib import Path

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
