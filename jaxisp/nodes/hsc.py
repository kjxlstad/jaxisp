
import jax.numpy as jnp
from jax import jit
from pydantic import validate_call


@validate_call
def hsc(
    hue_offset: int,
    saturation_gain: int,
    saturation_sdr: int,
):
    hue_offset = jnp.pi * hue_offset / 180
    sin_hue = (256 * jnp.sin(hue_offset)).astype(jnp.int32)
    cos_hue = (256 * jnp.cos(hue_offset)).astype(jnp.int32)

    def compute(array):
        cb_image, cr_image = jnp.split(array, 2, axis=-1)

        hsc_cb_image = jnp.right_shift(
            cos_hue * (cb_image - 128) - sin_hue * (cr_image - 128), 8)
        hsc_cb_image = jnp.right_shift(
            saturation_gain * hsc_cb_image, 8) + 128

        hsc_cr_image = jnp.right_shift(
            sin_hue * (cb_image - 128) + cos_hue * (cr_image - 128), 8)
        hsc_cr_image = jnp.right_shift(
            saturation_gain * hsc_cr_image, 8) + 128

        hsc_cbcr_image = jnp.stack([hsc_cb_image, hsc_cr_image], axis=-1)
        return jnp.clip(hsc_cbcr_image, 0, saturation_sdr)

    return jit(compute)