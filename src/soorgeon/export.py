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

from pathlib import Path

import jupytext
import yaml

from soorgeon import parse, static_analysis


class NotebookExporter:
    """Converts a notebook into a Ploomber pipeline
    """
    def __init__(self, nb):
        self._nb = nb

        self._proto_tasks = self._init_proto_tasks(nb)

        # snippets map names with the code the task will contain, we use
        # them to run static analysis
        self._snippets = {pt.name: str(pt) for pt in self._proto_tasks}

        self._io = None

    def _init_proto_tasks(self, nb):
        """Breask notebook into smaller sections
        """
        # use H2 headers to break notebook
        breaks = parse.find_breaks(nb)

        # generate groups of cells
        cells_split = parse.split_with_breaks(nb.cells, breaks)

        # extract names by using the H2 header text
        names = parse.names_with_breaks(nb.cells, breaks)

        # initialize proto tasks
        return [
            parse.ProtoTask(name, cell_group)
            for name, cell_group in zip(names, cells_split)
        ]

    def get_task_specs(self):
        """Return task specs (dictionary) for each proto task
        """
        return {pt.name: pt.to_spec(self.io) for pt in self._proto_tasks}

    def get_sources(self):
        """
        Generate the .py code strings (percent format) for each proto task
        """
        # FIXME: this calls find_providers, we should only call it once
        upstream = static_analysis.find_upstream(self._snippets)

        providers = static_analysis.ProviderMapping(self.io)

        code_nb = self._get_code()

        return {
            pt.name: pt.export(upstream, self._io, providers, code_nb)
            for pt in self._proto_tasks
        }

    def _get_code(self):
        return '\n'.join(cell['source'] for cell in self._nb.cells)

    @property
    def io(self):
        """
        {name: (inputs, outputs), ...}
        """
        if self._io is None:
            io = static_analysis.find_io(self._snippets)
            self._io = static_analysis.prune_io(io)

        return self._io


def from_nb(nb):
    exporter = NotebookExporter(nb)

    task_specs = exporter.get_task_specs()

    sources = exporter.get_sources()

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

    # TODO: instantiate dag since this may raise issues and we want to capture
    # them to let the user know how to fix them (e.g., more >1 H2 headers with
    # the same text)


def from_path(path):
    from_nb(jupytext.read(path))
