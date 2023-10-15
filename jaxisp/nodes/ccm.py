import jax.numpy as jnp
from jax import jit
from pydantic import validate_call

from jaxisp.type_utils import ImageRGB


@validate_call
def ccm(
    correction_matrix: list[list[int]],
    saturation_hdr: int, # TODO: fixme
):

    ccm = jnp.array(correction_matrix).T
    matrix = ccm[:3]
    bias = ccm[3].reshape(1, 1, 3)

    def compute(array: ImageRGB) -> ImageRGB:
        ccm_rgb_image = array @ matrix + bias
        return jnp.clip(ccm_rgb_image >> 10, 0, saturation_hdr)

    return jit(compute)
