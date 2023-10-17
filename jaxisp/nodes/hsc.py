from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call

from jaxisp.nodes.common import sdr_filter


@validate_call
def hsc[Input: Shaped[Array, "h w 2"], Output: Shaped[Array, "h w 2"]](
    hue_offset: int,
    saturation_gain: int,
    saturation_sdr: int,
) -> Callable[[Input], Output]:
    hue_offset = jnp.pi * hue_offset / 180
    sin_hue = (256 * jnp.sin(hue_offset)).astype(jnp.int32)
    cos_hue = (256 * jnp.cos(hue_offset)).astype(jnp.int32)

    @sdr_filter
    def compute(array: Input) -> Output:
        cb_image, cr_image = jnp.split(array, 2, axis=-1)

        hsc_cb_image = jnp.right_shift(
            cos_hue * (cb_image - 128) - sin_hue * (cr_image - 128), 8)
        hsc_cb_image = jnp.right_shift(
            saturation_gain * hsc_cb_image, 8) + 128

        hsc_cr_image = jnp.right_shift(
            sin_hue * (cb_image - 128) + cos_hue * (cr_image - 128), 8)
        hsc_cr_image = jnp.right_shift(
            saturation_gain * hsc_cr_image, 8) + 128

        hsc_cbcr_image = jnp.concatenate([hsc_cb_image, hsc_cr_image], axis=2)
        return jnp.clip(hsc_cbcr_image, 0, saturation_sdr)

    return jit(compute)
