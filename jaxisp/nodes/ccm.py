from typing import Callable

import jax.numpy as jnp
from jax import jit
from pydantic import validate_call

from jaxisp.nodes.common import hdr_filter
from jaxisp.type_utils import ImageRGB


@validate_call
def ccm[Input: ImageRGB, Output: ImageRGB](
    correction_matrix: list[list[int]],
    saturation_hdr: int,
) -> Callable[[Input], Output]:

    ccm = jnp.array(correction_matrix).T
    matrix = ccm[:3]
    bias = ccm[3].reshape(1, 1, 3)

    @hdr_filter
    def compute(array: Input) -> Output:
        ccm_rgb_image = array @ matrix + bias
        return jnp.clip(ccm_rgb_image >> 10, 0, saturation_hdr)

    return jit(compute)
