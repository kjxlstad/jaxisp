from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Int32, Shaped
from pydantic import validate_call

from jaxisp.helpers import bilateral_filter, gaussian_kernel
from jaxisp.nodes.common import sdr_filter


@jit
def calculate_intensity_lut(sigma_intensity: int) -> Shaped[Int32, "65025"]:
    diff = jnp.arange(255 ** 2)
    lut = 1024 * jnp.exp(-diff / (2.0 * (255 * sigma_intensity) ** 2))
    return lut.astype(jnp.int32)

@validate_call
def bnf[Input: Shaped[Array, 'h w 3'], Output: Shaped[Array, 'h w 3']](
    sigma_intensity: float,
    sigma_spatial: float,
) -> Callable[[Input], Output]:
    itensity_weight_lut = calculate_intensity_lut(sigma_intensity)
    spatial_weight = gaussian_kernel(sigma_spatial, kernel_size=5)
    spatial_weight = (
        1024 * spatial_weight / spatial_weight.max()).astype(jnp.int32)

    @sdr_filter
    def compute(array: Input) -> Output:
        bf_y_image = bilateral_filter(
            array, spatial_weight, itensity_weight_lut, right_shift=10)
        return bf_y_image

    return jit(compute)
