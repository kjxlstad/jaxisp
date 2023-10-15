from operator import call
from typing import Callable, TextIO

import yaml
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from returns.result import Failure, Success, safe

from jaxisp.helpers import BayerPattern


# TODO: add typing
class ISPNode:
    def __post_init__(self):
        self.execute = self.compile()

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def from_yaml(self, f: TextIO) -> "ISPNode":
        return self(**yaml.load(f, Loader=yaml.SafeLoader))


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
