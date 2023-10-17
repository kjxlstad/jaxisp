from typing import Any

import yaml
from jax import jit
from pydantic import field_validator
from pydantic.dataclasses import dataclass

from jaxisp import nodes
from jaxisp.nodes.common import SensorConfig

# TODO: should also have in- and out- types for each node
# figure out where this is best defined
VALID_NODES = {node.__name__.lower(): node for node in nodes.__all__}

def hdr_saturation(
    raw_sat: int,
    black_level_r: int,
    black_level_gr: int,
    black_level_gb: int,
    black_level_b: int,
    alpha: int,
    beta: int,
) -> int:
    hdr_max_r = raw_sat - black_level_r
    hdr_max_b = raw_sat - black_level_b
    hdr_max_gr = raw_sat - black_level_gr + hdr_max_r * alpha // 1024
    hdr_max_gb = raw_sat - black_level_gb + hdr_max_b * beta // 102
    return max(hdr_max_r, hdr_max_b, hdr_max_gr, hdr_max_gb)


@dataclass
class Pipeline:
    sensor: SensorConfig
    node_order: list[str]
    parameters: dict[str, Any]

    @field_validator("node_order")
    @classmethod
    def validate_node_order(cls, v):
        if not isinstance(v, list):
            raise TypeError("node_order must be a list")
        for node in v:
            if node not in VALID_NODES:
                raise ValueError(f"Invalid node name: {node}")
        return v

    def __post_init__(self):
        self.saturation_levels = self.calculate_saturation_levels()

        self.nodes = {
            node_name: self.compile_node(node_name, self.parameters[node_name])
            for node_name in self.node_order
        }

        def forward(array):
            for node_name, node in self.nodes.items():
                self.on_before_node(node_name)
                array = node(array)
                self.on_after_node(node_name)
            return array

        self.execute = jit(forward)

    def on_before_node(self, node_name: str):
        pass

    def on_after_node(self, node_name: str):
        pass

    def compile_node(self, node_name: str, node_params: dict[str, Any]):
        node = VALID_NODES[node_name]
        needed_fields = node.__annotations__.keys()

        if "sensor" in needed_fields:
            node_params["sensor"] = self.sensor

        for stage, value in self.saturation_levels.items():
            sat = f"saturation_{stage}"
            if sat in needed_fields:
                node_params[sat] = value

        # TODO: add logging of compilation times
        return node(**node_params)

    def calculate_saturation_levels(self) -> dict[str, int]:
        raw_sat = 2**self.sensor.bit_depth - 1
        sdr_sat = 255
        sat_levels = {"raw": raw_sat, "sdr": sdr_sat}

        if "blc" in self.node_order:
            hdr_sat = hdr_saturation(raw_sat, **self.parameters["blc"])
            return sat_levels | {"hdr": hdr_sat}

        return sat_levels | {"hdr": raw_sat}

    def __call__(self, array):
        return self.execute(array)

    def __repr__(self) -> str:
        return f"Pipeline({", ".join(self.node_order)})"

if __name__ == "__main__":
    with (
        open("jaxisp/configs/pipeline.yaml", "r") as config_file,
        open("jaxisp/configs/parameters/nikon_d3200.yaml") as param_file,
    ):
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        parameters = yaml.load(param_file, Loader=yaml.SafeLoader)
