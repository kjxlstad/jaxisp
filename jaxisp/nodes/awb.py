import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic.dataclasses import dataclass

from jaxisp.helpers import merge_bayer, split_bayer
from jaxisp.nodes.common import ISPNode, SensorConfig


@dataclass
class AWB(ISPNode):
    gain_r: int
    gain_gr: int
    gain_gb: int
    gain_b: int

    sensor: SensorConfig
    saturation_hdr: int # TODO: fixme

    def compile(self):
        def compute(
            bayer_mosaic: Shaped[Array, "h w"]
        ) -> Shaped[Array, "h w"]:
            channels = split_bayer(bayer_mosaic, self.sensor.bayer_pattern)
            gains = jnp.array(
                [self.gain_r, self.gain_gr, self.gain_gb, self.gain_b]
            ).reshape(4, 1, 1)

            wb_channels = (channels * gains) >> 10
            wb_bayer = merge_bayer(wb_channels, self.sensor.bayer_pattern)

            return jnp.clip(wb_bayer, 0, self.saturation_hdr)

        return jit(compute)
