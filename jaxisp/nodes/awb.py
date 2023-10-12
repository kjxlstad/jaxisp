from functools import partial

from jax import jit
import jax.numpy as jnp

from jaxisp.nodes.common import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer
from jaxisp.array_types import BayerMosaic


class AWB(ISPNode):
    def compile(
        self,
        bayer_pattern: str,
        r_gain: int,
        gr_gain: int,
        gb_gain: int,
        b_gain: int,
        saturation_hdr: int,
        **kwargs
    ):
        bayer_pattern = BayerPattern.from_str(bayer_pattern)

        bayer_to_channels = partial(split_bayer, pattern=bayer_pattern)
        channels_to_bayer = partial(merge_bayer, pattern=bayer_pattern)

        def compute(array: BayerMosaic) -> BayerMosaic:
            channels = bayer_to_channels(array)
            gains = jnp.array([r_gain, gr_gain, gb_gain, b_gain]).reshape(4, 1, 1)

            wb_channels = (channels * gains) >> 10
            wb_bayer = channels_to_bayer(wb_channels)
            return jnp.clip(wb_bayer, 0, saturation_hdr)

        return jit(compute)
