"""Dead Pixel Correction
TODO: add link to diagram
"""

from typing import Any
from functools import partial, reduce
from enum import Enum

from jax import jit, vmap
import jax.numpy as jnp

from jaxisp.nodes import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer, shift


class DPC(ISPNode):
    sensor_bayer_pattern: BayerPattern
    diff_threshold: int
    
    def compile(self, config: dict[str, Any]):
        bayer_pattern = BayerPattern[config["bayer_pattern"].upper()]
        diff_threshold = config["diff_threshold"]

        bayer_to_channels = partial(split_bayer, pattern=bayer_pattern)
        channels_to_bayer = partial(merge_bayer, pattern=bayer_pattern)
        rolling_window = jit(vmap(partial(shift, window_size=3), in_axes=0, out_axes=1))

        def compute(array):
            padded = jnp.pad(array, 2, mode="reflect")
            channels = bayer_to_channels(padded)
            offset = rolling_window(channels)
            
            center = offset[4]
            mask = reduce(
                jnp.multiply,
                (jnp.abs(center - offset[n]) > diff_threshold for n in set(range(9)) - {4})
            )

            diff_stack = jnp.stack([
                jnp.abs(2 * center - offset[1] - offset[7]),
                jnp.abs(2 * center - offset[3] - offset[5]),
                jnp.abs(2 * center - offset[0] - offset[8]),
                jnp.abs(2 * center - offset[6] - offset[2]),
            ], axis=-1)
            
            indices = jnp.argmin(diff_stack, axis=-1, keepdims=True)
            
            neighbor_stack = jnp.stack([
                offset[1] + offset[7],
                offset[3] + offset[5],
                offset[0] + offset[8],
                offset[6] + offset[2],
            ], axis=-1)
            
            dpc_array = jnp.take_along_axis(neighbor_stack >> 1, indices, axis=-1).squeeze(-1)
            res_array = mask * dpc_array + ~mask * center

            return channels_to_bayer(res_array)

        return jit(compute)


from jsonargparse import ArgumentParser
from jsonargparse import ActionConfigFile

def cli(node: ISPNode):
    annotations = node.__annotations__
    parser = ArgumentParser()
    from jsonargparse import CLI
    parser.add_argument("--config", action=ActionConfigFile)
    for name, type_ in annotations.items():
        parser.add_argument(f"--{name.replace("_", "-")}", type=type_, required=True)
    return parser

if __name__ == "__main__":
    # load image as some type
    # process image
    # save image as some type
    args = cli(DPC).parse_args()
    print(args)