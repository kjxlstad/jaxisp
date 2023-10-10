from functools import partial

from jax import jit, vmap
import jax.numpy as jnp

from jaxisp.nodes import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer, shift


class AAF(ISPNode):
    def compile(self, config):
        bayer_pattern = BayerPattern[config["bayer_pattern"].upper()]

        bayer_to_channels = partial(split_bayer, pattern=bayer_pattern)
        channels_to_bayer = partial(merge_bayer, pattern=bayer_pattern)
        rolling_window = jit(vmap(partial(shift, window_size=3), in_axes=0, out_axes=1))

        def compute(array):
            padded = jnp.pad(array, 2, mode="reflect")
            channels = bayer_to_channels(padded)
            shifted = rolling_window(channels)

            multipliers = jnp.ones_like(shifted).at[4].set(8)
            aaf_channels = (shifted * multipliers).sum(axis=0) >> 4

            return channels_to_bayer(aaf_channels)

        return jit(compute)
