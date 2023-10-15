from typing import Any

import yaml
from jax import jit
from pydantic.dataclasses import dataclass

from jaxisp import nodes
from jaxisp.nodes.common import SensorConfig

# TODO: should also have in- and out- types for each node
# figure out where this is best defined
VALID_NODES = {node.__name__.lower(): node for node in nodes.__all__}


@dataclass
class Pipeline:
    sensor: SensorConfig
    node_order: list[str]
    parameters: dict[str, Any]

    def compile_node(self, node_name: str, node_params: dict[str, Any]):
        node = VALID_NODES[node_name]
        needed_fields = node.__annotations__.keys()

        if "sensor" in needed_fields:
            node_params["sensor"] = self.sensor

        # FIXME: saturation should be calculated with a callback
        for stage, value in {"raw": 4095, "hdr": 1023, "sdr": 255}.items():
            sat = f"saturation_{stage}"
            if sat in needed_fields:
                node_params[sat] = value

        # TODO: add logging of compilation times
        return node(**node_params)

    def __post_init__(self):
        compiled_nodes = [
            self.compile_node(node_name, self.parameters[node_name])
            for node_name in self.node_order
        ]

        def compute(array):
            for node in compiled_nodes:
                array = node(array)
            return array

        self.execute = jit(compute)

    def __call__(self, array):
        return self.execute(array)
    
    def __repr__(self) -> str:
        return f"Pipeline({", ".join(self.node_order)})"


if __name__ == "__main__":
    with (open("jaxisp/configs/pipeline.yaml", "r") as config_file,
        open("jaxisp/configs/parameters/nikon_d3200.yaml") as param_file):
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        parameters = yaml.load(param_file, Loader=yaml.SafeLoader)
        pipeline = Pipeline(**config, parameters=parameters)
