import copy
import re

_IS_IPYTHON_CELL_MAGIC = r'^\s*%{2}[a-zA-Z]+'
_IS_IPYTHON_LINE_MAGIC = r'^\s*%{1}[a-zA-Z]+'
_IS_INLINE_SHELL = r'^\s*!{1}.+'

_IS_COMMENTED_LINE_MAGIC = r'^(.+) # \[magic\] (%.*)'

_PREFIX = '# [magic] '
_PREFIX_LEN = len(_PREFIX)

# these are magics that can modify the dependency structure beacuse they
# may declare new variables or use existing ones as inputs
HAS_INLINE_PYTHON = {'%%capture', '%%timeit', '%%time', '%time', '%timeit'}


def comment_magics(nb):
    """
    Iterates over cells, commenting the ones with magics
    """
    nb = copy.deepcopy(nb)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell['source'] = _comment_if_ipython_magic(cell['source'])

    return nb


def uncomment_magics(nb):
    """
    Iterates over cells, uncommenting the ones with magics
    """
    nb = copy.deepcopy(nb)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell['source'] = _uncomment_magics_cell(cell['source'])

    return nb


def _delete_magic(line):
    """Returns an empty line if it starts with the # [magic] prefix
    """
    return '' if line.startswith(_PREFIX) else line


def _delete_magics_cell(source):
    """Reverts the comments applied to magics (cell level)
    """
    if not source:
        return source

    lines = source.splitlines()
    lines_new = [_delete_magic(line) for line in lines]
    return '\n'.join(lines_new).strip()


def _uncomment_magic(line):
    """Reverts the comments applied to magics (line level)
    """
    if line.startswith(_PREFIX):
        return line[_PREFIX_LEN:]

    parts = _is_commented_line_magic(line)

    if parts:
        code, magic = parts
        return f'{magic} {code}'
    else:
        return line


def _uncomment_magics_cell(source):
    """Reverts the comments applied to magics (cell level)
    """
    lines = source.splitlines()
    lines_new = [_uncomment_magic(line) for line in lines]
    return '\n'.join(lines_new)


def _comment(line):
    """Adds the # [magic] prefix (line level)
    """
    return f'# [magic] {line}'


def _comment_ipython_line_magic(line, magic):
    """Adds suffix to line magics: # [magic] %{name}

    Converts:
    %timeit x = 1

    Into:
    x = 1 # [magic] %timeit
    """
    return line.replace(magic, '').strip() + f' # [magic] {magic.strip()}'


def _comment_if_ipython_magic(source):
    """Comments lines into comments if they're IPython magics (cell level)
    """
    # TODO: support for nested cell magics. e.g.,
    # %%timeit
    # %%timeit
    # something()
    lines_out = []
    comment_rest = False

    # TODO: inline magics should add a comment at the end of the line, because
    # the python code may change the dependency structure. e.g.,
    # %timeit z = x + y -> z = x + y # [magic] %timeit
    # note that this only applies to inline magics that take Python code as arg

    # NOTE: magics can take inputs but their outputs ARE NOT saved. e.g.,
    # %timeit x = y + 1
    # running such magic requires having y but after running it, x IS NOT
    # declared. But this is magic dependent %time x = y + 1 will add x to the
    # scope

    for line in source.splitlines():
        cell_magic = _is_ipython_cell_magic(line)

        if comment_rest:
            lines_out.append(_comment(line))
        else:
            line_magic = _is_ipython_line_magic(line)

            # if line magic, comment line
            if line_magic:
                # NOTE: switched _comment_ipython_line_magic(line, line_magic)
                # for _comment
                lines_out.append(_comment(line))

            # if inline shell, comment line
            elif _is_inline_shell(line):
                lines_out.append(_comment(line))

            # if cell magic, comment line
            elif cell_magic in HAS_INLINE_PYTHON:
                lines_out.append(_comment(line))

            # if cell magic whose content *is not* Pytho, comment line and
            # all the remaining lines in the cell
            elif cell_magic:
                lines_out.append(_comment(line))
                comment_rest = True

            # otherwise, don't do anything
            else:
                lines_out.append(line)

    return '\n'.join(lines_out)


# NOTE: the code in the following lines is based on Ploomber's source code.
# We did that instead of importing it since Ploomber has many dependencies. At
# some point, we may integrate this into Ploomber and remove this.


def _is_commented_line_magic(source):
    """Determines if the source is an IPython cell magic. e.g.,

    %cd some-directory
    """
    m = re.match(_IS_COMMENTED_LINE_MAGIC, source)

    if not m:
        return False

    return m.group(1), m.group(2)


def _is_ipython_cell_magic(source):
    """Determines if the source is an IPython cell magic. e.g.,

    %cd some-directory
    """
    m = re.match(_IS_IPYTHON_CELL_MAGIC, source.lstrip())

    if not m:
        return False

    return m.group()


def _is_ipython_line_magic(line):
    """
    Determines if the source line is an IPython magic. e.g.,

    %%bash
    for i in 1 2 3; do
      echo $i
    done
    """
    m = re.match(_IS_IPYTHON_LINE_MAGIC, line)

    if not m:
        return False

    return m.group()


def _is_inline_shell(line):
    m = re.match(_IS_INLINE_SHELL, line)

    if not m:
        return False

    return m.group()
