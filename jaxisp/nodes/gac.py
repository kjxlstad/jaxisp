from typing import Callable

import jax.numpy as jnp
from jax import jit
from pydantic import validate_call

from jaxisp.type_utils import ImageRGB


@validate_call
def gac[Input: ImageRGB, Output: ImageRGB](
    gain: int,
    gamma: float,
    saturation_sdr: int,
    saturation_hdr: int,
) -> Callable[[Input], Output]:
    x = jnp.arange(saturation_hdr + 1)
    lut = (((x / saturation_hdr) ** gamma) * saturation_sdr).astype(jnp.uint8)

    def compute(array: Input) -> Output:
        gac_rgb_image = (array * gain) >> 8
        gac_rgb_image = jnp.clip(gac_rgb_image, 0, saturation_hdr)
        return lut[gac_rgb_image]

    return jit(compute)
