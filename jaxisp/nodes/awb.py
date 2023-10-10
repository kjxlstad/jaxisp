from functools import partial

from jax import jit, vmap
import jax.numpy as jnp

from jaxisp.nodes import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer, shift

class AWB(ISPNode):
    def compile(self, config):
        bayer_pattern = BayerPattern[config["bayer_pattern"].upper()]
        r_gain = config["r_gain"]
        gr_gain = config["gr_gain"]
        gb_gain = config["gb_gain"]
        b_gain = config["b_gain"]
        saturation_values_hdr = config["saturation_values_hdr"]
        
        bayer_to_channels = partial(split_bayer, pattern=bayer_pattern)
        channels_to_bayer = partial(merge_bayer, pattern=bayer_pattern)
        
        def compute(array):
            channels = bayer_to_channels(array)
            gains = jnp.array([[[r_gain]], [[gr_gain]], [[gb_gain]], [[b_gain]]])
            
            wb_channels = (channels * gains) >> 10
            wb_bayer = channels_to_bayer(wb_channels)
            return jnp.clip(wb_bayer, 0, saturation_values_hdr)
        
        return jit(compute)