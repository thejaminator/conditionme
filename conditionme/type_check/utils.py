from typing import NoReturn, Optional, TypeVar

A = TypeVar("A")


def should_not_happen(msg: str = "This should not happen") -> NoReturn:
    raise AssertionError(msg)


def assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)


def assert_not_none(value: Optional[A], message: str = "Value should not be None") -> A:
    assert value is not None, message
    return value
