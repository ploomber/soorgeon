import typing as t
from click.exceptions import ClickException
from click._compat import get_text_stderr
from click.utils import echo
from gettext import gettext as _


def _format_message(exception):
    if hasattr(exception, 'format_message'):
        return exception.format_message()
    else:
        return str(exception)


def _build_message(exception):
    msg = _format_message(exception)

    while exception.__cause__:
        msg += f'\n{_format_message(exception.__cause__)}'
        exception = exception.__cause__

    return msg


class BaseException(ClickException):
    """
    A subclass of ClickException that adds support for printing error messages
    from chained exceptions
    """

    def show(self, file: t.Optional[t.IO] = None) -> None:
        if file is None:
            file = get_text_stderr()

        message = _build_message(self)
        echo(_("Error: {message}").format(message=message), file=file)


class InputWontRunError(BaseException):
    """Raised when there are errors that make running the input infeasible
    """
    pass


class InputError(BaseException):
    """Raised when the input has issues and needs user's editing
    """
    pass


class InputSyntaxError(InputWontRunError):
    """Raised if the notebook has invalid syntax
    """
    pass
