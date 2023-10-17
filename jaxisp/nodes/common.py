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


def filter_type(in_types: Iterable[type], out_types: Iterable[type]):
    """Creates a decorator that handles safe input and output type casting."""
    def decorator(func):
        def wrapper(*arrays):
            inputs = [a.astype(t) for a, t in zip(arrays, in_types)]
            result = func(*inputs)

            if isinstance(result, jnp.ndarray):
                return result.astype(out_types)

            return tuple(a.astype(t) for a, t in zip(result, out_types))
        return wrapper
    return decorator


raw_filter = filter_type(jnp.int32, jnp.uint16)
hdr_filter = filter_type(jnp.int32, jnp.uint16)
sdr_filter = filter_type(jnp.int32, jnp.uint8)
