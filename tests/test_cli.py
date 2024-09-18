from unittest.mock import Mock
from pathlib import Path

import yaml
import jupytext
import jupytext.formats
import pytest
import shutil
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


@pytest.mark.parametrize(
    "args, product_prefix",
    [
        [["nb.py"], "output"],
        [["nb.py", "--product-prefix", "another"], "another"],
        [["nb.py", "-p", "another"], "another"],
    ],
)
def test_refactor_product_prefix(tmp_empty, args, product_prefix):
    Path("nb.py").write_text(simple)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    spec = DAGSpec("pipeline.yaml")

    paths = [
        i for product in [t["product"].values() for t in spec["tasks"]] for i in product
    ]

    assert result.exit_code == 0
    assert all([p.startswith(product_prefix) for p in paths])


@pytest.mark.parametrize(
    "input_, out_ext, args",
    [
        ["nb.py", "py", ["nb.py"]],
        ["nb.ipynb", "ipynb", ["nb.ipynb"]],
        ["nb.py", "ipynb", ["nb.py", "--file-format", "ipynb"]],
        ["nb.ipynb", "py", ["nb.ipynb", "--file-format", "py"]],
    ],
)
def test_refactor_file_format(tmp_empty, input_, out_ext, args):
    jupytext.write(jupytext.reads(simple, fmt="py:light"), input_)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    assert result.exit_code == 0

    # test the output file has metadata, otherwise it may fail to execute
    # if missing the kernelspec info
    assert jupytext.read(Path("tasks", f"cell-0.{out_ext}")).metadata
    assert jupytext.read(Path("tasks", f"cell-2.{out_ext}")).metadata
    assert jupytext.read(Path("tasks", f"cell-4.{out_ext}")).metadata


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


@pytest.mark.parametrize(
    "args, ext, requirements",
    [
        [["nb.py"], "pkl", "ploomber>=0.14.7"],
        [["nb.py", "--df-format", "parquet"], "parquet", "ploomber>=0.14.7\npyarrow"],
        [["nb.py", "--df-format", "csv"], "csv", "ploomber>=0.14.7"],
    ],
    ids=[
        "none",
        "parquet",
        "csv",
    ],
)
@pytest.mark.parametrize(
    "nb, products_expected",
    [
        [
            simple,
            [
                "output/cell-0-x.pkl",
                "output/cell-0.ipynb",
                "output/cell-2-y.pkl",
                "output/cell-2.ipynb",
                "output/cell-4.ipynb",
            ],
        ],
        [
            with_dfs,
            [
                "output/first-df.{ext}",
                "output/first.ipynb",
                "output/second.ipynb",
            ],
        ],
        [
            mixed,
            [
                "output/first-x.pkl",
                "output/first-df.{ext}",
                "output/first.ipynb",
                "output/second.ipynb",
            ],
        ],
    ],
    ids=[
        "simple",
        "with-dfs",
        "mixed",
    ],
)
def test_refactor_df_format(tmp_empty, args, ext, nb, products_expected, requirements):
    Path("nb.py").write_text(nb)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    spec = DAGSpec("pipeline.yaml")

    paths = [
        i for product in [t["product"].values() for t in spec["tasks"]] for i in product
    ]

    assert result.exit_code == 0
    assert set(paths) == set(p.format(ext=ext) for p in products_expected)

    content = "# Auto-generated file" f", may need manual editing\n{requirements}\n"
    assert Path("requirements.txt").read_text() == content


with_dfs = """\
# ## first

df = 1

# ## second

df_2 = df + 1

"""

with_lambda = """\
# ## first

num_square = lambda n: n**2

# ## second

num_square = num_square(2)

"""


@pytest.mark.parametrize(
    "args, requirements",
    [
        [["nb.py", "--serializer", "cloudpickle"], "cloudpickle\nploomber>=0.14.7"],
        [["nb.py", "--serializer", "dill"], "dill\nploomber>=0.14.7"],
    ],
    ids=["cloudpickle", "dill"],
)
@pytest.mark.parametrize(
    "nb, products_expected",
    [
        [
            with_dfs,
            [
                "output/first-df.pkl",
                "output/first.ipynb",
                "output/second.ipynb",
            ],
        ],
        [
            with_lambda,
            [
                "output/first-num_square.pkl",
                "output/first.ipynb",
                "output/second.ipynb",
                "output/second-num_square.pkl",
            ],
        ],
    ],
    ids=["with-dfs", "with-lambda"],
)
def test_refactor_serializer(tmp_empty, args, nb, products_expected, requirements):
    Path("nb.py").write_text(nb)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    spec = DAGSpec("pipeline.yaml")

    paths = [
        i for product in [t["product"].values() for t in spec["tasks"]] for i in product
    ]
    assert set(paths) == set(products_expected)
    assert result.exit_code == 0

    content = "# Auto-generated file" f", may need manual editing\n{requirements}\n"
    assert Path("requirements.txt").read_text() == content


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


@pytest.mark.parametrize(
    "nb, requirements",
    [
        [imports_pyarrow, "ploomber>=0.14.7\npyarrow"],
        [imports_fastparquet, "fastparquet\nploomber>=0.14.7"],
        [imports_nothing, "ploomber>=0.14.7\npyarrow"],
    ],
    ids=[
        "pyarrow",
        "fastparquet",
        "nothing",
    ],
)
def test_refactor_parquet_requirements(tmp_empty, nb, requirements):
    Path("nb.py").write_text(nb)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ["nb.py", "--df-format", "parquet"])

    assert result.exit_code == 0
    content = "# Auto-generated file" f", may need manual editing\n{requirements}\n"
    assert Path("requirements.txt").read_text() == content


@pytest.mark.parametrize(
    "input_, backup, file_format, source",
    [
        ["nb.ipynb", "nb-backup.ipynb", [], "nb.ipynb"],
        ["nb.py", "nb-backup.py", [], "nb.py"],
        ["nb.ipynb", "nb-backup.ipynb", ["--file-format", "py"], "nb.py"],
        ["nb.py", "nb-backup.py", ["--file-format", "ipynb"], "nb.ipynb"],
    ],
)
def test_single_task(tmp_empty, input_, backup, file_format, source):
    jupytext.write(jupytext.reads(simple, fmt="py:light"), input_)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, [input_, "--single-task"] + file_format)

    assert result.exit_code == 0

    with Path("pipeline.yaml").open() as f:
        spec = yaml.safe_load(f)

    assert spec == {
        "tasks": [
            {
                "source": source,
                "product": "products/nb-report.ipynb",
            }
        ]
    }

    # test the output file has metadata, otherwise it may fail to execute
    # if missing the kernelspec info
    assert jupytext.read(Path(source)).metadata
    assert jupytext.read(Path(backup)).metadata


@pytest.mark.parametrize(
    "code",
    [
        """
# ## header

if something
    pass
""",
        """
# ## header

y = x + 1
""",
    ],
    ids=[
        "syntax-error",
        "undefined-name",
    ],
)
def test_doesnt_suggest_single_task_if_nb_cannot_run(tmp_empty, code):
    Path("nb.py").write_text(code)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ["nb.py"])

    assert result.exit_code == 1
    assert "soorgeon refactor nb.py --single-task" not in result.output


@pytest.mark.parametrize(
    "code",
    [
        """
from math import *
""",
        """
y = 1

def x():
    return y
""",
        """
x = 1
""",
    ],
    ids=[
        "star-import",
        "fn-with-global-vars",
        "missing-h2-heading",
    ],
)
def test_doesnt_suggest_single_task_if_nb_can_run(tmp_empty, code):
    Path("nb.py").write_text(code)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ["nb.py"])

    assert result.exit_code == 1
    assert "soorgeon refactor nb.py --single-task" in result.output


def test_suggests_single_task_if_export_crashes(tmp_empty, monkeypatch):
    monkeypatch.setattr(export.NotebookExporter, "export", Mock(side_effect=KeyError))

    Path("nb.py").write_text(simple)

    runner = CliRunner()
    result = runner.invoke(cli.refactor, ["nb.py"])

    assert result.exit_code == 1
    assert "soorgeon refactor nb.py --single-task" in result.output


# adds import if needed / and doesn't add import pickle


def test_clean_py(tmp_empty):
    Path("nb.py").write_text(simple)

    runner = CliRunner()
    runner.invoke(cli.refactor, ["nb.py"])
    result = runner.invoke(cli.clean, ["tasks/cell-2.py"])
    assert result.exit_code == 0
    # black
    assert "Reformatted tasks/cell-2.py with black" in result.output
    # end of basic_clean()
    assert "Finished cleaning tasks/cell-2.py" in result.output


def test_clean_ipynb(tmp_empty):
    nb_ = jupytext.reads(simple, fmt="py:light")
    jupytext.write(nb_, "nb.ipynb")

    runner = CliRunner()
    runner.invoke(cli.refactor, ["nb.ipynb"])
    result = runner.invoke(cli.clean, ["tasks/cell-2.ipynb"])

    assert result.exit_code == 0
    # black
    assert "Reformatted tasks/cell-2.ipynb with black" in result.output
    # end of basic_clean()
    assert "Finished cleaning tasks/cell-2.ipynb" in result.output


@pytest.mark.parametrize(
    "content, fmt",
    [
        [
            """
```python
import soorgeon
import atexit

s = 'something'
```
""",
            "markdown",
        ],
        [
            """\
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---

```{code-cell}
import soorgeon
import atexit

s = 'something'
```
""",
            "myst",
        ],
    ],
    ids=[
        "md",
        "myst",
    ],
)
def test_clean_markdown(tmp_empty, content, fmt):
    Path("file.md").write_text(content)

    runner = CliRunner()
    result = runner.invoke(cli.clean, ["file.md"])

    assert result.exit_code == 0
    # black
    assert "Reformatted file.md with black." in result.output
    # end of basic_clean()
    assert "Finished cleaning file.md" in result.output

    metadata = jupytext.formats.read_metadata(Path("file.md").read_text(), "md")

    if metadata:
        fmt_read = metadata["jupytext"]["text_representation"]["format_name"]
        assert fmt_read == fmt


@pytest.mark.parametrize(
    "name, content",
    [
        ["file.py", "import math"],
        [
            "file.md",
            """
```python
import math
```
""",
        ],
    ],
    ids=[
        "py",
        "md",
    ],
)
def test_lint(tmp_empty, name, content):
    Path(name).write_text(content)

    runner = CliRunner()
    result = runner.invoke(cli.lint, [name])
    assert result.exit_code == 0
    assert name in result.output
    assert "F401 'math' imported" in result.output


def test_clean_no_task(tmp_empty):
    nb_ = jupytext.reads(simple, fmt="py:light")
    jupytext.write(nb_, "nb.ipynb")

    runner = CliRunner()
    runner.invoke(cli.refactor, ["nb.ipynb"])
    result = runner.invoke(cli.clean, ["tasks/cell-9.ipynb"])

    assert result.exit_code == 2
    assert "Error: Invalid value for 'FILENAME'" in result.output


output_test = """\
# ## first

import fastparquet
from pathlib import Path
import pandas as pd

j = open('file.txt', 'w')
j.close()

df = pd.DataFrame()
df.to_csv('file_csv.csv')
df.to_parquet('my.parquet')
Path('write_text').write_text("stf")
Path('write_byte').write_bytes('stf')


"""

output_with_comment_test = """\
# ## second

import fastparquet
from pathlib import Path
import pandas as pd

# j = open('file.txt', 'w')
# j.close()

df = pd.DataFrame()
df.to_csv('file_csv.csv')
Path('write_text').write_text("stf")

f = open('tmp.txt')
k = open('tmp.txt', 'r')

'''
Path('write_byte').write_bytes('stf')
df.to_parquet('my.parquet')
'''
"""


def test_refactor_product_should_warning_if_notebook_output_file(tmp_empty):
    Path("nb.py").write_text(output_test)
    args = "nb.py"

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    assert result.exit_code == 0
    assert "open" in result.output
    assert "to_csv" in result.output
    assert "to_parquet" in result.output
    assert "write_text" in result.output
    assert "write_bytes" in result.output


def test_refactor_product_should_not_warning_if_comment(tmp_empty):
    Path("nb.py").write_text(output_with_comment_test)
    args = "nb.py"

    runner = CliRunner()
    result = runner.invoke(cli.refactor, args)

    assert result.exit_code == 0
    assert "open" not in result.output
    assert "to_csv" in result.output
    assert "to_parquet" not in result.output
    assert "write_text" in result.output
    assert "write_bytes" not in result.output


ModuleNotFoundError_sample = """
# ## header

import nomodule
# ## header

"""

AttributeError_sample = """
# ## header

import math
print(math.logg(1))
# ## header
"""

SyntaxError_sample = """
# ## header

impor math
print(math.log(1))
# ## header
"""

OtherError_sample = """
# ## header

import math
print(math.log(-5))
# ## header
"""


@pytest.mark.parametrize(
    "code, output",
    [
        [simple, "no error encountered"],
        [ModuleNotFoundError_sample, "packages are missing, please install them"],
        [AttributeError_sample, "might be due to changes in the libraries"],
        [SyntaxError_sample, "There are syntax errors in the notebook"],
    ],
)
def test_test_notebook_runs(tmp_empty, code, output):
    nb_ = jupytext.reads(code, fmt="py:light")
    filenames = ["nb.ipynb", "nb.py"]
    output_paths = ["nb-output.ipynb", None]
    for filename in filenames:
        for output_path in output_paths:
            shutil.rmtree(str(output_path), ignore_errors=True)
            jupytext.write(nb_, filename)
            runner = CliRunner()
            if output_path:
                expected_output_path = output_path
                result = runner.invoke(cli.test, [filename, output_path])
            else:
                expected_output_path = "nb-soorgeon-test.ipynb"
                result = runner.invoke(cli.test, [filename])
            if output == "no error encountered":
                assert result.exit_code == 0
            else:
                assert result.exit_code == 1
            assert output in result.output
            assert Path(expected_output_path).exists()
