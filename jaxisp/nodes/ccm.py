import jax.numpy as jnp
from jax import jit
from pydantic.dataclasses import dataclass

from jaxisp.nodes.common import ISPNode
from jaxisp.type_utils import ImageRGB


@dataclass
class CCM(ISPNode):
    correction_matrix: list[list[int]]

    saturation_hdr: int # TODO: fixme

    def compile(self):
        ccm = jnp.array(self.correction_matrix).T
        matrix = ccm[:3]
        bias = ccm[3].reshape(1, 1, 3)

        def compute(array: ImageRGB) -> ImageRGB:
            ccm_rgb_image = array @ matrix + bias
            return jnp.clip(ccm_rgb_image >> 10, 0, self.saturation_hdr)

        return jit(compute)
