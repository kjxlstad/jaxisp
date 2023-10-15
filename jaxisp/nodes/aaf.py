import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic.dataclasses import dataclass

from jaxisp.helpers import bayer_neighbor_pixels, merge_bayer
from jaxisp.nodes.common import ISPNode, SensorConfig


# TODO: this is actually slower than numpy implementation
@dataclass
class AAF(ISPNode):
    sensor: SensorConfig

    def compile(self):
        def compute(
            bayer_mosaic: Shaped[Array, "h w"]
        ) -> Shaped[Array, "h w"]:
            grid = bayer_neighbor_pixels(
                bayer_mosaic, self.sensor.bayer_pattern
            )

            multipliers = jnp.ones_like(grid).at[4].set(8)
            aaf_channels = (grid * multipliers).sum(axis=0) >> 4

            return merge_bayer(aaf_channels, self.sensor.bayer_pattern)

        return jit(compute)
