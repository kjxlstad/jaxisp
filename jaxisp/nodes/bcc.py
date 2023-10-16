from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call


@validate_call
def bcc[Input: Shaped[Array, 'h w 3'], Output: Shaped[Array, 'h w 3']](
    brightness_offset: int,
    contrast_gain: int,
    saturation_sdr: int,
) -> Callable[[Input], Output]:
    brightness_offset = jnp.array(brightness_offset).astype(jnp.int32)
    contrast_gain = jnp.array(contrast_gain).astype(jnp.int32)

    def compute(array: Input) -> Output:
        bcc_y_image = jnp.clip(array + brightness_offset, 0, saturation_sdr)

        y_median = jnp.median(bcc_y_image).astype(jnp.int32)
        bcc_y_image = jnp.right_shift(
            (bcc_y_image - y_median) * contrast_gain, 8) + y_median
        bcc_y_image = jnp.clip(bcc_y_image, 0, saturation_sdr)

        return array

    return jit(compute)
