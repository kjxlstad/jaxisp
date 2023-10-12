from jax import jit
import jax.numpy as jnp

from jaxisp.nodes.common import ISPNode


class CCM(ISPNode):
    def compile(
        self, 
        ccm: list[list[int]],
        saturation_hdr: int,
        **kwarg
    ):
        ccm = jnp.array(ccm).T
        matrix = ccm[:3]
        bias = ccm[3].reshape(1, 1, 3)

        def compute(array):
            ccm_rgb_image = array @ matrix + bias
            return jnp.clip(ccm_rgb_image >> 10, 0, saturation_hdr)

        return jit(compute)
