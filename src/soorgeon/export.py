"""
High-level description of the refactoring process

Given some notebook:

```python
# H1 header

# H2 header
x = 1 + 1

# H2 header
y = x + 1

# H3 header
z = x + 1
```

Step 1. Section splitting.

We split it into sections using H2 markdown headers (H1 goes into the first
task), we call this a proto-task:


```python
# H1 header

# H2 header
x = 1 + 1
```

```python
# H2 header
y = x + 1
```

```python
# H3 header
z = x + 1
```

Step 2. Parse inputs and outputs.

Then we figure out which variables each proto-task uses and which ones they
create:

1. Exposes `x`
2. Exposes `y`, uses `x`
3. Exposes `z`, uses `x`


Step 3. Resolve dependencies.

Then we resolve dependencies:

1 -> 2
1 -> 3


Step 4. Determine variables to serialize.

Now we determine the products for each proto-task, ignoring the variables that
aren't used downstream:

1. x
2. None
3. None


Step 5. Add serialization cell.

Then, we modify each task to serialize the outputs we need:

```python
# added by soorgeon
from pathlib import Path
import pickle

# H1 header

# H2 header
x = 1 + 1

# added by soorgeon
Path(product['x']).write_bytes(x)
```

Step 6. Add parameters cell with upstram dependencies.

We also add the parameters cell with the upstream dependdencies:

```python
# %% tags=["parameters"]
upstream = ['one']

# H2 header
y = x + 1
```

Step 7. Refactor imports.

We also need to modify imports, we parse all imports in the
notebooks, delete them, and then add a new cell at the top with the
subset of import statements that the proto task uses.

```python
# soorgeon auto-generated imports statements
import pandas as pd

# %% tags=["parameters"]
upstream = ['one']

# H2 header
y = x + 1
```

Step 8. Generate step.

Finally, we generate the pipeline.yaml file.
"""
import shutil
import traceback
import ast
import pprint
from collections import namedtuple
from pathlib import Path
import logging
from importlib import resources
from soorgeon import assets

import click
import parso
import jupytext
import yaml
import nbformat

from soorgeon import (split, io, definitions, proto, exceptions, magics,
                      pyflakes)

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)


class NotebookExporter:
    """Converts a notebook into a Ploomber pipeline
    """
    def __init__(self, nb, verbose=True, df_format=None, py=False):
        if df_format not in {None, 'parquet', 'csv'}:
            raise ValueError("df_format must be one of "
                             "None, 'parquet' or 'csv', "
                             f"got: {df_format!r}")

        # NOTE: we're commenting magics here but removing them in ProtoTask,
        # maybe we should comment magics also in ProtoTask?
        nb = magics.comment_magics(nb)

        self._nb = nb
        self._df_format = df_format
        self._verbose = verbose

        self._io = None
        self._definitions = None
        self._tree = None
        self._providers = None

        self._check()

        self._proto_tasks = self._init_proto_tasks(nb, py)

        # snippets map names with the code the task will contain, we use
        # them to run static analysis
        self._snippets = {pt.name: str(pt) for pt in self._proto_tasks}

    def export(self, product_prefix=None):
        """Export the project

        Parameters
        ---------
        product_prefix : str
            A prefix to append to all products. If None, it is set to 'output'
        """
        product_prefix = product_prefix or 'output'

        # export functions and classes to a separate file
        self.export_definitions()

        # export requirements.txt
        self.export_requirements()

        # export .gitignore
        self.export_gitignore(product_prefix)

        task_specs = self.get_task_specs(product_prefix=product_prefix)

        sources = self.get_sources()

        dag_spec = {'tasks': list(task_specs.values())}

        for name, task_spec in task_specs.items():
            path = Path(task_spec['source'])
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(sources[name])

        out = yaml.dump(dag_spec, sort_keys=False)
        # pyyaml doesn't have an easy way to control whitespace, but we want
        # tasks to have an empty line between them
        out = out.replace('\n- ', '\n\n- ')

        Path('pipeline.yaml').write_text(out)

        self.export_readme()

    def _check(self):
        """
        Run a few checks before continuing the refactoring. If this fails,
        we'll require the user to do some small changes to their code.
        """
        code = self._get_code()
        _check_syntax(code)
        pyflakes.check_notebook(self._nb)
        _check_functions_do_not_use_global_variables(code)
        _check_no_star_imports(code)

    def _init_proto_tasks(self, nb, py):
        """Break notebook into smaller sections
        """
        # use H2 headers to break notebook
        breaks = split.find_breaks(nb)

        # generate groups of cells
        cells_split = split.split_with_breaks(nb.cells, breaks)

        # extract names by using the H2 header text
        names = split.names_with_breaks(nb.cells, breaks)

        # initialize proto tasks
        return [
            proto.ProtoTask(
                name,
                cell_group,
                df_format=self._df_format,
                py=py,
            ) for name, cell_group in zip(names, cells_split)
        ]

    def get_task_specs(self, product_prefix=None):
        """Return task specs (dictionary) for each proto task
        """
        return {
            pt.name: pt.to_spec(self.io, product_prefix=product_prefix)
            for pt in self._proto_tasks
        }

    def get_sources(self):
        """
        Generate the code strings (ipynb or percent format) for each proto task
        """
        # FIXME: this calls find_providers, we should only call it once
        upstream = io.find_upstream(self._snippets)

        code_nb = self._get_code()

        return {
            pt.name: pt.export(
                upstream,
                self.io,
                self.providers,
                code_nb,
                self.definitions,
            )
            for pt in self._proto_tasks
        }

    def export_definitions(self):
        """Create an exported.py file with function and class definitions
        """
        # do not create exported.py if there are no definitions
        if not self.definitions:
            return

        out = '\n\n'.join(self.definitions.values())

        ip = io.ImportsParser(self._get_code())
        imports = ip.get_imports_cell_for_task(out)

        if imports:
            exported = f'{imports}\n\n\n{out}'
        else:
            exported = out

        Path('exported.py').write_text(exported)

    def export_requirements(self):
        """Generates requirements.txt file, appends it at the end if already
        exists
        """
        reqs = Path('requirements.txt')

        # ploomber is added by default (pinned to >=0.14.7 because earlier
        # versions throw an error when using the inline bash IPython magic
        # during the static_analysis stage)
        pkgs = ['ploomber>=0.14.7'] + definitions.packages_used(self.tree)

        # add pyarrow to requirements if needed
        if (self._df_format == 'parquet' and 'pyarrow' not in pkgs
                and 'fastparquet' not in pkgs):
            pkgs = ['pyarrow'] + pkgs

        pkgs_txt = '\n'.join(sorted(pkgs))

        out = f"""\
# Auto-generated file, may need manual editing
{pkgs_txt}
"""
        if reqs.exists():
            reqs.write_text(reqs.read_text() + out)
        else:
            reqs.write_text(out)

    def _get_code(self):
        """Returns the source of code cells
        """
        return '\n'.join(cell['source'] for cell in self._nb.cells
                         if cell['cell_type'] == 'code')

    def export_gitignore(self, product_prefix):
        if product_prefix and not Path(product_prefix).is_absolute():
            path = Path('.gitignore')
            content = '' if not path.exists() else path.read_text() + '\n'
            path.write_text(content + product_prefix + '\n')
            self._echo(f'Added {str(product_prefix)!r} directory'
                       ' to .gitignore...')

    def export_readme(self):
        path = Path('README.md')

        if path.exists():
            content = path.read_text() + '\n'
            self._echo('README.md found, appended auto-generated content')
        else:
            content = ''
            self._echo('Added README.md')

        path.write_text(content + resources.read_text(assets, 'README.md'))

    def _echo(self, msg):
        if self._verbose:
            click.echo(msg)

    @property
    def definitions(self):
        if self._definitions is None:
            self._definitions = (definitions.from_def_and_class(self.tree))

        return self._definitions

    @property
    def tree(self):
        if self._tree is None:
            code = self._get_code()
            self._tree = parso.parse(code)

        return self._tree

    @property
    def providers(self):
        if self._providers is None:
            self._providers = io.ProviderMapping(self.io)

        return self._providers

    @property
    def io(self):
        """
        {name: (inputs, outputs), ...}
        """
        if self._io is None:
            io_ = self._get_raw_io()

            logging.info(f'io: {pp.pformat(io_)}\n')

            self._io = io.prune_io(io_)

            logging.info(f'pruned io: {pp.pformat(self._io)}\n')

        return self._io

    def _get_raw_io(self):
        return io.find_io(self._snippets)


FunctionNeedsFix = namedtuple('FunctionNeedsFix', ['name', 'pos', 'args'])


def _check_syntax(code):
    try:
        ast.parse(code)
    except SyntaxError:
        error = traceback.format_exc()
    else:
        error = None

    if error:
        raise exceptions.InputSyntaxError(f'Could not refactor notebook due '
                                          f'to invalid syntax\n\n {error}')


def _check_no_star_imports(code):
    tree = parso.parse(code)

    star_imports = [
        import_ for import_ in tree.iter_imports() if import_.is_star_import()
    ]

    if star_imports:
        star_imports_ = '\n'.join(import_.get_code()
                                  for import_ in star_imports)
        url = ('https://github.com/ploomber/soorgeon/blob/main/doc'
               '/star-imports.md')
        raise exceptions.InputError(
            'Star imports are not supported, please change '
            f'the following:\n\n{star_imports_}\n\n'
            f'For more details, see: {url}')


# see issue #12 on github
def _check_functions_do_not_use_global_variables(code):
    tree = parso.parse(code)

    needs_fix = []

    local_scope = set(definitions.find_defined_names(tree))

    for funcdef in tree.iter_funcdefs():
        # FIXME: this should be passing the tree directly, no need to reparse
        # again, but for some reason,
        # using find_inputs_and_outputs_from_tree(funcdef) returns the name
        # of the function as an input
        in_, _ = io.find_inputs_and_outputs(funcdef.get_code(),
                                            local_scope=local_scope)

        if in_:
            needs_fix.append(
                FunctionNeedsFix(
                    funcdef.name.value,
                    funcdef.start_pos,
                    in_,
                ))

    if needs_fix:
        message = ('Looks like the following functions are using global '
                   'variables, this is unsupported. Please add all missing '
                   'arguments. See this to learn more:\n'
                   'https://github.com/ploomber/soorgeon/blob'
                   '/main/doc/fn-global.md\n\n')

        def comma_separated(args):
            return ','.join(f"'{arg}'" for arg in args)

        message += '\n'.join(
            f'* Function {f.name!r} uses variables {comma_separated(f.args)}'
            for f in needs_fix)

        raise exceptions.InputError(message)


def from_nb(nb, log=None, product_prefix=None, df_format=None, py=False):
    """Refactor a notebook by passing a notebook object

    Parameters
    ----------
    product_prefix : str
        A prefix to add to all products. If None, it's set to 'output'

    """
    if log:
        logging.basicConfig(level=log.upper())

    exporter = NotebookExporter(nb, df_format=df_format, py=py)

    exporter.export(product_prefix=product_prefix)

    # TODO: instantiate dag since this may raise issues and we want to capture
    # them to let the user know how to fix them (e.g., more >1 H2 headers with
    # the same text)


def from_path(path, log=None, product_prefix=None, df_format=None, py=False):
    """Refactor a notebook by passing a path to it

    Parameters
    ----------
    allow_single_task : bool
        If False, the function will fail if it cannot refactor the notebook
        into a multi-stage pipeline. If True, it will first try to refactor
        the notebook, and if it fails, it will generate a pipeline with
        a single task
    """
    from_nb(jupytext.read(path),
            log=log,
            product_prefix=product_prefix,
            df_format=df_format,
            py=py)


def single_task_from_path(path, product_prefix, file_format):
    """Refactor a notebook into a single task Ploomber pipeline
    """
    path = Path(path)

    click.echo('Creating a pipeline with a single task...')

    cell = nbformat.v4.new_code_cell(source='upstream = None',
                                     metadata=dict(tags=['parameters']))

    nb = jupytext.read(path)
    nb.cells.insert(0, cell)

    name = path.stem
    path_backup = path.with_name(f'{name}-backup{path.suffix}')

    # output
    ext = path.suffix[1:] if file_format is None else file_format
    path_to_task = f'{name}.{ext}'

    # create backup
    shutil.copy(path, path_backup)

    jupytext.write(nb,
                   path_to_task,
                   fmt='py:percent' if ext == 'py' else 'ipynb')

    spec = {
        'tasks': [{
            'source':
            path_to_task,
            'product':
            str(Path(product_prefix or 'products', f'{name}-report.ipynb'))
        }]
    }

    pipeline = 'pipeline.yaml'
    click.echo(f'Done. Copied code to {path_to_task!r} and added it to '
               f'{pipeline!r}. Created backup of original notebook '
               f'at {str(path_backup)!r}.')

    Path('pipeline.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))


def refactor(path, log, product_prefix, df_format, single_task, file_format):

    if single_task:
        single_task_from_path(path=path,
                              product_prefix=product_prefix,
                              file_format=file_format)
    else:
        ext = Path(path).suffix[1:] if file_format is None else file_format

        try:
            from_nb(jupytext.read(path),
                    log=log,
                    product_prefix=product_prefix,
                    df_format=df_format,
                    py=ext == 'py')
        # InputError means the input is broken
        except exceptions.InputWontRunError:
            raise
        # This implies an error on our end
        except Exception as e:
            logger.exception('Error calling from_nb')
            cmd = f'soorgeon refactor {path} --single-task'
            msg = ('An error occurred when refactoring '
                   'notebook.\n\nTry refactoring '
                   f'as a single task pipeline:\n\n$ {cmd}\n\n'
                   'Need help? https://ploomber.io/community\n\n'
                   'Error details:\n')
            raise exceptions.InputError(msg) from e
