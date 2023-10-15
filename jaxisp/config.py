from operator import call
from typing import Callable, TextIO

import yaml
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from returns.result import Failure, Success, safe

from jaxisp.helpers import BayerPattern, CFAMode


def parse[I, O](field: str, parse_fun: Callable[[I], O], mode: str = "before"):
    @field_validator(field, mode=mode, check_fields=False)
    def validate(cls, value: I) -> str | O:
        match call(safe(lambda: parse_fun(value))):
            case Success(pattern):    return pattern
            case Failure(KeyError()): return value
    return validate


@dataclass
class YamlConfig:
    @classmethod
    def from_yaml(cls, f: TextIO) -> "YamlConfig":
        return cls(**yaml.load(f, Loader=yaml.SafeLoader))


@dataclass
class DPCParams:
    diff_threshold: int


@dataclass
class BLCParams:
    black_level_r: int
    black_level_gr: int
    black_level_gb: int
    black_level_b: int
    alpha: int
    beta: int


@dataclass
class AWBParams:
    gain_r: int
    gain_gr: int
    gain_gb: int
    gain_b: int


@dataclass
class CNFParams:
    diff_threshold: int
    gain_r: int
    gain_b: int


@dataclass
class CFAParams:
    mode: CFAMode


@dataclass
class CCMParams:
    correction_matrix: list[list[int]]


@dataclass
class GACParams:
    gain: int
    gamma: float


@dataclass
class NLMParams:
    window_size: int
    patch_size: int
    h: int


@dataclass
class BNFParams:
    sigma_intensity: float
    sigma_spatial: float


@dataclass
class CEHParams:
    tiles: list[int]
    clip: float


@dataclass
class EEHParams:
    edge_gain: float
    edge_threshold: int
    flat_threshold: int
    delta_threshold: int


@dataclass
class FCSParams:
    delta_min: int
    delta_max: int


@dataclass
class HSCParams:
    hue_offset: int
    saturation_gain: int


@dataclass
class BCCParams:
    brightness_offset: int
    contrast_gain: int


@dataclass
class SCLParams:
    width: int
    height: int


@dataclass
class Parameters(YamlConfig):
    dpc: DPCParams
    blc: BLCParams
    awb: AWBParams
    cnf: CNFParams
    cfa: CFAParams
    ccm: CCMParams
    gac: GACParams
    nlm: NLMParams
    bnf: BNFParams
    ceh: CEHParams
    eeh: EEHParams
    fcs: FCSParams
    hsc: HSCParams
    bcc: BCCParams
    scl: SCLParams


@dataclass
class SensorConfig(YamlConfig):
    width: int
    height: int
    bit_depth: int
    bayer_pattern: BayerPattern

    parse_pattern = parse(
        "bayer_pattern", lambda v: BayerPattern[v.upper()]
    )


@dataclass
class PipelineConfig(YamlConfig):
    sensor: SensorConfig


if __name__ == "__main__":
    with open("jaxisp/configs/parameters/nikon_d3200.yaml") as f:
        params = Parameters.from_yaml(f)
        print(params)

    with open("jaxisp/configs/pipeline.yaml") as f:
        pipeline = PipelineConfig.from_yaml(f)
        print(pipeline)
