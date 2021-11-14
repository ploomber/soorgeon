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
        return {pt.name: pt.to_spec(self.io) for pt in self._proto_tasks}

    def get_sources(self):
        """
        Generate the .py code strings (precent format) for each proto task
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
            self._io = static_analysis.find_io(self._snippets)

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

    Path('pipeline.yaml').write_text(yaml.dump(dag_spec))

    # TODO: instantiate dag since this may raise issues and we want to capture
    # them to let the user know how to fix them (e.g., more >1 H2 headers with
    # the same text)


def from_path(path):
    from_nb(jupytext.read(path))
