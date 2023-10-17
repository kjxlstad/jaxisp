from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call

from jaxisp.helpers import bayer_neighbor_pixels, merge_bayer
from jaxisp.nodes.common import SensorConfig, raw_filter


# TODO: this is actually slower than numpy implementation
@validate_call
def aaf[Input: Shaped[Array, "h w"], Output: Shaped[Array, "h w"]](
    sensor: SensorConfig
) -> Callable[[Input], Output]:
    @raw_filter
    def compute(bayer_mosaic: Input) -> Output:
        grid = bayer_neighbor_pixels(
            bayer_mosaic, sensor.bayer_pattern
        )

        multipliers = jnp.ones_like(grid).at[4].set(8)
        aaf_channels = (grid * multipliers).sum(axis=0) >> 4

        return merge_bayer(aaf_channels, sensor.bayer_pattern)

    return jit(compute)
