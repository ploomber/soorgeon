"""
NOTE: this was taken from Ploomber, we copied it instead of importing it
since ploomber has many dependencies, at some point we'll either move
this functionality into soorgeon or create a ploomber-core package
and move things over there
"""
import warnings
from io import StringIO

from pyflakes import api as pyflakes_api
from pyflakes.reporter import Reporter
from pyflakes.messages import (UndefinedName, UndefinedLocal,
                               DuplicateArgument, ReturnOutsideFunction,
                               YieldOutsideFunction, ContinueOutsideLoop,
                               BreakOutsideLoop)

from soorgeon.exceptions import InputWontRunError, InputSyntaxError

# messages: https://github.com/PyCQA/pyflakes/blob/master/pyflakes/messages.py
_ERRORS = (
    UndefinedName,
    UndefinedLocal,
    DuplicateArgument,
    ReturnOutsideFunction,
    YieldOutsideFunction,
    ContinueOutsideLoop,
    BreakOutsideLoop,
)


def _process_messages(mesages):
    return '\n'.join(str(msg) for msg in mesages)


def process_errors_and_warnings(messages):
    errors, warnings = [], []

    for message in messages:
        if isinstance(message, _ERRORS):
            errors.append(message)
        else:
            warnings.append(message)

    return _process_messages(errors), _process_messages(warnings)


# https://github.com/PyCQA/pyflakes/blob/master/pyflakes/reporter.py
class MyReporter(Reporter):

    def __init__(self):
        self._stdout = StringIO()
        self._stderr = StringIO()
        self._stdout_raw = []
        self._unexpected = False
        self._syntax = False

    def flake(self, message):
        self._stdout_raw.append(message)
        self._stdout.write(str(message))
        self._stdout.write('\n')

    def unexpectedError(self, *args, **kwargs):
        """pyflakes calls this when ast.parse raises an unexpected error
        """
        self._unexpected = True
        return super().unexpectedError(*args, **kwargs)

    def syntaxError(self, *args, **kwargs):
        """pyflakes calls this when ast.parse raises a SyntaxError
        """
        self._syntax = True
        return super().syntaxError(*args, **kwargs)

    def _seek_zero(self):
        self._stdout.seek(0)
        self._stderr.seek(0)

    def _make_error_message(self, error):
        return ('Errors detected in your source code:'
                f'\n{error}\n\n'
                '(ensure that your notebook executes from top-to-bottom '
                'and try again)')

    def _check(self):
        self._seek_zero()

        # syntax errors are stored in _stderr
        # https://github.com/PyCQA/pyflakes/blob/master/pyflakes/api.py

        error_message = '\n'.join(self._stderr.readlines())

        if self._syntax:
            raise InputSyntaxError(self._make_error_message(error_message))
        elif self._unexpected:
            warnings.warn('An unexpected error happened '
                          f'when analyzing code: {error_message.strip()!r}')
        else:
            errors, warnings_ = process_errors_and_warnings(self._stdout_raw)

            if warnings_:
                warnings.warn(warnings_)

            if errors:
                raise InputWontRunError(self._make_error_message(errors))


def check_notebook(nb):
    """
    Run pyflakes on a notebook, wil catch errors such as missing passed
    parameters that do not have default values

    Parameters
    ----------
    nb : NotebookNode
        Notebook object. Must have a cell with the tag "parameters"

    filename : str
        Filename to identify pyflakes warnings and errors

    Raises
    ------
    SyntaxError
        If the notebook's code contains syntax errors

    TypeError
        If params and nb do not match (unexpected or missing parameters)

    RenderError
        When certain pyflakes errors are detected (e.g., undefined name)
    """
    # concatenate all cell's source code in a single string
    source_code = '\n'.join(c['source'] for c in nb.cells
                            if c.cell_type == 'code')

    # this objects are needed to capture pyflakes output
    reporter = MyReporter()

    # run pyflakes.api.check on the source code
    pyflakes_api.check(source_code, filename='', reporter=reporter)

    reporter._check()
