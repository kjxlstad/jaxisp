from jax import jit
import jax.numpy as jnp

from jaxisp.nodes.common import ISPNode


class GAC(ISPNode):
    def compile(
        self,
        gain: int,
        gamma: int,
        saturation_hdr: int,
        saturation_sdr: int,
        **kwargs
    ):
        x = jnp.arange(saturation_hdr + 1)
        lut = ((x / saturation_hdr) ** gamma) * saturation_sdr

        def compute(array):
            gac_rgb_image = (array * gain) >> 8
            gac_rgb_image = jnp.clip(gac_rgb_image, 0, saturation_hdr)
            return lut[gac_rgb_image]

        return jit(compute)