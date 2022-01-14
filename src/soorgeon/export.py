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
import ast
import pprint
from collections import namedtuple
from pathlib import Path

import logging
import parso
import jupytext
import yaml

from soorgeon import split, io, definitions, proto

logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)


class NotebookExporter:
    """Converts a notebook into a Ploomber pipeline
    """

    def __init__(self, nb):
        self._nb = nb

        self._check()

        self._proto_tasks = self._init_proto_tasks(nb)

        # snippets map names with the code the task will contain, we use
        # them to run static analysis
        self._snippets = {pt.name: str(pt) for pt in self._proto_tasks}

        self._io = None
        self._definitions = None

    def export(self, product_prefix=None):
        """Export the project
        """
        # export functions and classes to a separate file
        self.export_definitions()

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

    def _check(self):
        """
        Run a few checks before continuing the refactoring. If this fails,
        we'll require the user to do some small changes to their code.
        """
        code = self._get_code()
        _check_syntax(code)
        _check_functions_do_not_use_global_variables(code)

    def _init_proto_tasks(self, nb):
        """Breask notebook into smaller sections
        """
        # use H2 headers to break notebook
        breaks = split.find_breaks(nb)

        # generate groups of cells
        cells_split = split.split_with_breaks(nb.cells, breaks)

        # extract names by using the H2 header text
        names = split.names_with_breaks(nb.cells, breaks)

        # initialize proto tasks
        return [
            proto.ProtoTask(name, cell_group)
            for name, cell_group in zip(names, cells_split)
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
        Generate the .py code strings (percent format) for each proto task
        """
        # FIXME: this calls find_providers, we should only call it once
        upstream = io.find_upstream(self._snippets)

        providers = io.ProviderMapping(self.io)

        code_nb = self._get_code()

        return {
            pt.name: pt.export(upstream, self._io, providers, code_nb,
                               self.definitions)
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

    def _get_code(self):
        """Returns the source of code cells
        """
        return '\n'.join(cell['source'] for cell in self._nb.cells
                         if cell['cell_type'] == 'code')

    @property
    def definitions(self):
        if self._definitions is None:
            code = self._get_code()
            tree = parso.parse(code)
            self._definitions = (definitions.from_def_and_class(tree))

        return self._definitions

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
    except SyntaxError as e:
        raise SyntaxError('Error refactoring notebook: invalid syntax') from e


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
        # FIXME: add a short guide and print url as part of the error message
        message = ('Looks like the following functions are using global '
                   'variables, this is unsupported. Please add all missing '
                   'arguments.\n\n')

        def comma_separated(args):
            return ','.join(f"'{arg}'" for arg in args)

        message += '\n'.join(
            f'* Function {f.name!r} uses variables {comma_separated(f.args)}'
            for f in needs_fix)

        raise ValueError(message)


def from_nb(nb, log=None, product_prefix=None):
    """

    Parameters
    ----------
    product_prefix : str
        A prefix to add to all products. If None, it's set to 'output'
    """
    if log:
        logging.basicConfig(level=log.upper())

    exporter = NotebookExporter(nb)
    exporter.export(product_prefix=product_prefix)

    # TODO: instantiate dag since this may raise issues and we want to capture
    # them to let the user know how to fix them (e.g., more >1 H2 headers with
    # the same text)


def from_path(path, log=None, product_prefix=None):
    from_nb(jupytext.read(path), log=log, product_prefix=product_prefix)
