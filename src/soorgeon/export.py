from pathlib import Path

import jupytext
import yaml

from soorgeon import parse, static_analysis


def from_nb(nb):
    breaks = parse.find_breaks(nb)
    cells_split = parse.split_with_breaks(nb.cells, breaks)

    names = parse.names_with_breaks(nb.cells, breaks)

    proto_tasks = [
        parse.ProtoTask(name, cell_group)
        for name, cell_group in zip(names, cells_split)
    ]

    snippets = {pt.name: str(pt) for pt in proto_tasks}

    # FIXME: this calls find_providers, we should only call it once
    upstream = static_analysis.find_upstream(snippets)
    io = static_analysis.find_io(snippets)
    providers = static_analysis.find_providers(io)

    sources = {
        pt.name: pt.export(upstream, io, providers)
        for pt in proto_tasks
    }
    task_specs = {pt.name: pt.to_spec(io) for pt in proto_tasks}

    dag_spec = {'tasks': list(task_specs.values())}

    for name, task_spec in task_specs.items():
        path = Path(task_spec['source'])
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(sources[name])

    Path('pipeline.yaml').write_text(yaml.dump(dag_spec))


def from_path(path):
    from_nb(jupytext.read(path))