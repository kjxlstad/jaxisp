import jax.numpy as jnp
from jax import jit
from pydantic.dataclasses import dataclass

from jaxisp.nodes.common import ISPNode
from jaxisp.type_utils import ImageRGB


@dataclass
class GAC(ISPNode):
    gain: int
    gamma: float

    saturation_sdr: int
    saturation_hdr: int

    def compile(self):
        x = jnp.arange(self.saturation_hdr + 1)
        lut = ((x / self.saturation_hdr) ** self.gamma) * self.saturation_sdr

        def compute(array: ImageRGB) -> ImageRGB:
            gac_rgb_image = (array * self.gain) >> 8
            gac_rgb_image = jnp.clip(gac_rgb_image, 0, self.saturation_hdr)
            return lut[gac_rgb_image]

        return jit(compute)
