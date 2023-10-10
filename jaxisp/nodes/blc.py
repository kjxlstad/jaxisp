from functools import partial

from jax import jit
import jax.numpy as jnp

from jaxisp.nodes import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer

class BLC(ISPNode):
    def compile(self, config):
        bayer_pattern = BayerPattern[config["bayer_pattern"].upper()]
        alpha = config["alpha"]
        beta = config["beta"]
        bl_r = config["bl_r"]
        bl_b = config["bl_b"]
        bl_gr = config["bl_gr"]
        bl_gb = config["bl_gb"]
        
        bayer_to_channels = partial(split_bayer, pattern=bayer_pattern)
        channels_to_bayer = partial(merge_bayer, pattern=bayer_pattern)
        
        def compute(array):
            r, gr, gb, b = bayer_to_channels(array)
            
            r = jnp.clip(r - bl_r, 0)
            b = jnp.clip(b - bl_b, 0)
            gr -= bl_gr - jnp.right_shift(r * alpha, 10)
            gb -= bl_gb - jnp.right_shift(b * beta, 10)
            
            return channels_to_bayer(jnp.stack([r, gr, gb, b]))
                    
        return jit(compute)
