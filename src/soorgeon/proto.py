"""
ProtoTask handles the logic to convert a notebook section into a Ploomber task
"""
from copy import deepcopy
from pathlib import Path

import nbformat
import jupytext
from jinja2 import Template

from soorgeon import io, magics

_PICKLING_TEMPLATE = Template("""\
{%- for product in products -%}
{%- if product.startswith('df') and df_format in ('parquet', 'csv') -%}
Path(product['{{product}}']).parent.mkdir(exist_ok=True, parents=True)
{{product}}.to_{{df_format}}(product['{{product}}'], index=False)
{%- else -%}
Path(product['{{product}}']).parent.mkdir(exist_ok=True, parents=True)
Path(product['{{product}}']).write_bytes(pickle.dumps({{product}}))
{%- endif %}

{% endfor -%}\
""")

_UNPICKLING_TEMPLATE = Template("""\
{%- for up, key in up_and_in -%}
{%- if key.startswith('df') and df_format in ('parquet', 'csv') -%}
{{key}} = pd.read_{{df_format}}(upstream['{{up}}']['{{key}}'])
{%- else -%}
{{key}} = pickle.loads(Path(upstream['{{up}}']['{{key}}']).read_bytes())
{%- endif %}
{% endfor -%}\
""")


def _new_pickling_cell(outputs, df_format):
    df_format = df_format or ''
    source = _PICKLING_TEMPLATE.render(products=sorted(outputs),
                                       df_format=df_format).strip()
    return nbformat.v4.new_code_cell(source=source)


def _new_unpickling_cell(up_and_in, df_format):
    df_format = df_format or ''
    source = _UNPICKLING_TEMPLATE.render(up_and_in=sorted(up_and_in,
                                                          key=lambda t:
                                                          (t[0], t[1])),
                                         df_format=df_format).strip()
    return nbformat.v4.new_code_cell(source=source)


class ProtoTask:
    """A group of cells that will be converted into a Ploomber task
    """

    def __init__(self, name, cells, df_format, py):
        self._name = name
        self._cells = cells
        self._df_format = df_format
        self._py = py

    @property
    def name(self):
        return self._name

    def exposes(self):
        """Return a list of variables that this prototask creates
        """
        pass

    def uses(self):
        """Return a list of variables that this prototask uses
        """
        pass

    def _pickling_cell(self, io):
        """Add cell that pickles the outputs
        """
        _, outputs = io[self.name]

        if outputs:
            pickling = _new_pickling_cell(outputs, self._df_format)
            pickling.metadata['tags'] = ['soorgeon-pickle']

            return pickling
        else:
            return None

    def _unpickling_cell(self, io, providers):
        """Add cell that unpickles the inputs
        """
        inputs, _ = io[self.name]

        if inputs:
            up_and_in = [(providers.get(input_, self.name), input_)
                         for input_ in inputs]

            unpickling = _new_unpickling_cell(up_and_in, self._df_format)
            unpickling.metadata['tags'] = ['soorgeon-unpickle']

            return unpickling
        else:
            return None

    def _add_parameters_cell(self, cells, upstream):
        """Add parameters cell at the top
        """
        source = ''

        upstream_current = upstream[self.name]

        if upstream_current:
            source += f'upstream = {list(upstream_current)}\n'
        else:
            source += 'upstream = None\n'

        source += 'product = None'

        parameters = nbformat.v4.new_code_cell(source=source)
        parameters.metadata['tags'] = ['parameters']

        return [parameters] + cells

    def _add_imports_cell(self, code_nb, add_pathlib_and_pickle, definitions,
                          df_format):
        # FIXME: instatiate this in the constructor so we only build it once
        ip = io.ImportsParser(code_nb)

        source_raw = ip.get_imports_cell_for_task(io.remove_imports(str(self)))

        # since we analyze the code structure as a code string (and not a
        # notebook) - if a magic (which is turned into a comment at this
        # stage) is right above an import statement, parso will make it part
        # of the import node, so we remove them here. Other comments are
        # unchanged
        source = magics._delete_magics_cell(source_raw)

        # FIXME: only add them if they're not already there
        if add_pathlib_and_pickle:
            source = source or ''
            source += '\nfrom pathlib import Path'
            source += '\nimport pickle'

        # only add them if unserializing or serializing
        if df_format in {'parquet', 'csv'}:
            source += '\nimport pandas as pd'

        if definitions:
            names = ', '.join(definitions)
            source = source or ''
            source += f'\nfrom exported import {names}'

        if source:
            cell = nbformat.v4.new_code_cell(source=source)
            cell.metadata['tags'] = ['soorgeon-imports']
            return cell

    def export(
        self,
        upstream,
        io_,
        providers,
        code_nb,
        definitions,
    ):
        """Export as a Python string

        Parameters
        ----------
        definitions : dict
            {name: code, ...} mapping with all the function and class
            definitions in the notebook. Used to add an import statement
            to the task
        """

        nb = nbformat.v4.new_notebook()
        # TODO: simplify, make each function return a single cell and then join
        # here

        cells = deepcopy(self._cells)

        # remove import statements from code cells
        # FIXME: remove function definitions and class definitions
        for cell in cells:
            if cell.cell_type == 'code':
                cell['source'] = io.remove_imports(cell['source'])

        # remove empty cells and whitespace-only cells (we may have some after
        # removing imports)
        cells = [cell for cell in cells if cell['source'].strip()]

        cell_unpickling = self._unpickling_cell(io_, providers)

        if cell_unpickling:
            cells = [cell_unpickling] + cells

        cells = self._add_parameters_cell(cells, upstream)

        cell_pickling = self._pickling_cell(io_)

        if cell_pickling:
            cells = cells + [cell_pickling]

        cell_imports = self._add_imports_cell(
            code_nb,
            add_pathlib_and_pickle=cell_pickling or cell_unpickling,
            definitions=definitions,
            df_format=self._df_format)

        pre = [cell_imports] if cell_imports else []

        nb.cells = pre + cells

        # TODO: H2 header should be the top cell

        # remove magics
        nb_out = magics.uncomment_magics(nb)

        # jupytext does not write metadata automatically when writing to
        # ipnyb (but it does for py:percent), we save it here to ensure
        # ipynb has the kernelspec info
        if not self._py:
            nb_out.metadata.kernelspec = {
                "display_name": 'Python 3',
                "language": 'python',
                "name": 'python3',
            }

        return jupytext.writes(nb_out,
                               fmt='py:percent' if self._py else 'ipynb')

    def to_spec(self, io, product_prefix):
        """

        Parameters
        ----------
        product_prefix : str
            A prefix to add to all products
        """
        _, outputs = io[self.name]

        # prefix products by name to guarantee they're unique
        products = {
            out: str(
                Path(product_prefix,
                     _product_name(self.name, out, self._df_format)))
            for out in outputs
        }

        # FIXME: check that there isn't an nb key already
        products['nb'] = str(Path(product_prefix, f'{self.name}.ipynb'))

        ext = '.py' if self._py else '.ipynb'

        return {
            'source': str(Path('tasks', self.name + ext)),
            'product': products
        }

    def __str__(self):
        """Retun the task as string (only code cells)
        """
        return '\n'.join(cell['source'] for cell in self._cells
                         if cell.cell_type == 'code')


def _product_name(task, variable, df_format):
    ext = ('pkl'
           if not df_format or not variable.startswith('df') else df_format)
    return f'{task}-{variable}.{ext}'
