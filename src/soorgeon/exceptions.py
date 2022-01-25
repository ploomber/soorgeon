from click.exceptions import ClickException


class InputError(ClickException):
    """Raised when the input has issues and needs user's editing
    """
    pass
