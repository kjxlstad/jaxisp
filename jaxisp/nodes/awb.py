import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call

from jaxisp.helpers import merge_bayer, split_bayer
from jaxisp.nodes.common import SensorConfig


@validate_call
def awb(
    gain_r: int,
    gain_gr: int,
    gain_gb: int,
    gain_b: int,

    sensor: SensorConfig,
    saturation_hdr: int, # TODO: fixme
):
    def compute(
        bayer_mosaic: Shaped[Array, "h w"]
    ) -> Shaped[Array, "h w"]:
        channels = split_bayer(bayer_mosaic, sensor.bayer_pattern)
        gains = jnp.array(
            [gain_r, gain_gr, gain_gb, gain_b]
        ).reshape(4, 1, 1)

        wb_channels = (channels * gains) >> 10
        wb_bayer = merge_bayer(wb_channels, sensor.bayer_pattern)

        return jnp.clip(wb_bayer, 0, saturation_hdr)

    return jit(compute)
