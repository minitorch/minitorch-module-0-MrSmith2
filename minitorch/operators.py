"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List, TypeVar

# ## Task 0.1


# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    return d / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

T = TypeVar("T")
U = TypeVar("U")




def map(fn: Callable[[T], U], xs: Iterable[T]) -> List[U]:
    return [fn(x) for x in xs]


def zipWith(fn: Callable[[T, U], U], xs: Iterable[T], ys: Iterable[U]) -> List[U]:
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(fn: Callable[[T, T], T], xs: Iterable[T]) -> T:
    it = iter(xs)
    try:
        result = next(it)
    except StopIteration:
        raise TypeError()
    for x in it:
        result = fn(result, x)
    return result


def negList(xs: Iterable[float]) -> List[float]:
    return map(lambda x: -x, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    return zipWith(lambda x, y: x + y, xs, ys)


def sum(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return reduce(lambda a, b: a + b, xs)


def prod(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 1.0
    return reduce(lambda a, b: a * b, xs)

