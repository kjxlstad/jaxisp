from operator import call
from typing import Callable, Iterable

import jax.numpy as jnp
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from returns.result import Failure, Success, safe

from jaxisp.helpers import BayerPattern

# TODO: add a wrapper that incorporated pydantic's validate_call

def parse[I, O](field: str, parse_fun: Callable[[I], O], mode: str = "before"):
    @field_validator(field, mode=mode, check_fields=False)
    def validate(cls, value: I) -> str | O:
        match call(safe(lambda: parse_fun(value))):
            case Success(pattern):    return pattern
            case Failure(KeyError()): return value
    return validate


@dataclass
class SensorConfig:
    width: int
    height: int
    bit_depth: int
    bayer_pattern: BayerPattern

    parse_pattern = parse(
        "bayer_pattern", lambda v: BayerPattern[v.upper()]
    )


def one_to_one_filter(in_type: type, out_type: type):
    """Creates a decorator that handles safe input and output type casting.
    For filters that take a single array as input and output.
    """
    def decorator(func):
        def wrapper(array):
            input_ = array.astype(in_type)
            return func(input_).astype(out_type)
        return wrapper
    return decorator

def one_to_many_filter(in_type: type, out_types: Iterable[type]):
    """Creates a decorator that handles safe input and output type casting.
    For filters that take a single array as input and output multiple arrays.
    """
    def decorator(func):
        def wrapper(array):
            input_ = array.astype(in_type)
            result = func(input_)
            return tuple(a.astype(t) for a, t in zip(result, out_types))
        return wrapper
    return decorator

def many_to_one_filter(in_types: Iterable[type], out_type: type):
    """Creates a decorator that handles safe input and output type casting.
    For filters that take multiple arrays as input and output a single array.
    """
    def decorator(func):
        def wrapper(*arrays):
            inputs = [a.astype(t) for a, t in zip(arrays, in_types)]
            result = func(*inputs)
            return result.astype(out_type)
        return wrapper
    return decorator

raw_filter = one_to_one_filter(jnp.int32, jnp.uint16)
hdr_filter = one_to_one_filter(jnp.int32, jnp.uint16)
sdr_filter = one_to_one_filter(jnp.int32, jnp.uint8)
