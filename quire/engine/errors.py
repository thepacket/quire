class QuireError(Exception):
    """Base class for user-facing errors. The message is shown verbatim in the worksheet."""


class ParseError(QuireError):
    pass


class UnitError(QuireError):
    pass


class EvalError(QuireError):
    pass
