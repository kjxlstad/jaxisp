from jax import jit
import jax.numpy as jnp

from jaxtyping import Array, Shaped

from jaxisp.nodes.common import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer


class AWB(ISPNode):
    def compile(
        self, bayer_pattern: str, r_gain: int, gr_gain: int, gb_gain: int, b_gain: int, saturation_hdr: int, **kwargs
    ):
        bayer_pattern = BayerPattern.from_str(bayer_pattern)

        def compute(bayer_mosaic: Shaped[Array, ["h w"]]) -> Shaped[Array, ["h w"]]:
            channels = split_bayer(bayer_mosaic, pattern=bayer_pattern)
            gains = jnp.array([r_gain, gr_gain, gb_gain, b_gain]).reshape(4, 1, 1)

            wb_channels = (channels * gains) >> 10
            wb_bayer = merge_bayer(wb_channels, pattern=bayer_pattern)
            return jnp.clip(wb_bayer, 0, saturation_hdr)

        return jit(compute)
