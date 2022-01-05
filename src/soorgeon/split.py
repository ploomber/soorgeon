"""
Functions for splitting a notebook file into smaller parts
"""
import string
import re

from soorgeon import exceptions


def find_breaks(nb):
    """Find index breaks based on H2 markdown indexes

    Notes
    -----
    The first element of the returned list may be >0 if the first H2 header
    isn't in the first cell
    """
    breaks = []

    # TODO: this should return named tuples with index and extracted names
    for idx, cell in enumerate(nb.cells):
        # TODO: more robust H2 detector
        if cell.cell_type == 'markdown' and _get_h2_header(cell.source):
            breaks.append(idx)

    if not breaks:
        raise exceptions.InputError('Expected to have at least one markdown '
                                    'with a level 2 header')

    return breaks


def split_with_breaks(cells, breaks):
    """Split a list of cells at given indexes

    Notes
    -----
    Given that the first index has the cell indx of the first H2 header, but
    there may be code in ealier cells, we ignore it. The first split is always
    from 0 to breaks[1]
    """
    breaks = breaks + [None]
    breaks[0] = 0

    cells_split = []

    for left, right in zip(breaks, breaks[1:]):
        cells_split.append(cells[left:right])

    return cells_split


def names_with_breaks(cells, breaks):
    return [_get_h2_header(cells[break_]['source']) for break_ in breaks]


def _sanitize_name(name):
    """Sanitize content of an H2 header to be used as a filename
    """
    # replace all non-aplhanumeric with a dash
    sanitized = re.sub('[^0-9a-zA-Z]+', '-', name.lower())

    # argo does not allow names to start with a digit when using dependencies
    if sanitized[0] in string.digits:
        sanitized = 'section-' + sanitized

    return sanitized


def _get_h2_header(md):
    lines = md.splitlines()

    found = None

    for line in lines:
        match = re.search(r'\s*##\s+(.+)', line)

        if match:
            found = _sanitize_name(match.group(1))

            break

    return found
