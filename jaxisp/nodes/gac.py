import jax.numpy as jnp
from jax import jit
from pydantic import validate_call

from jaxisp.type_utils import ImageRGB


@validate_call
def gac(
    gain: int,
    gamma: float,
    saturation_sdr: int,
    saturation_hdr: int,
):
    x = jnp.arange(saturation_hdr + 1)
    lut = ((x / saturation_hdr) ** gamma) * saturation_sdr

    def compute(array: ImageRGB) -> ImageRGB:
        gac_rgb_image = (array * gain) >> 8
        gac_rgb_image = jnp.clip(gac_rgb_image, 0, saturation_hdr)
        return lut[gac_rgb_image]

    return jit(compute)
