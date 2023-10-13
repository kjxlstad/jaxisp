from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Shaped

from jaxisp.nodes.common import ISPNode
from jaxisp.helpers import BayerPattern, bayer_neighbor_pixels, merge_bayer


class AAF(ISPNode):
    def compile(
        self,
        bayer_pattern: str,
        **kwargs,
    ):
        bayer_pattern = BayerPattern[bayer_pattern.upper()]

        def compute(bayer_mosaic: Shaped[Array, "h w"]) -> Shaped[Array, "4 h/2 w/2"]:
            grid = bayer_neighbor_pixels(bayer_mosaic, pattern=bayer_pattern)

            multipliers = jnp.ones_like(grid).at[4].set(8)
            aaf_channels = (grid * multipliers).sum(axis=0) >> 4

            return merge_bayer(aaf_channels, pattern=bayer_pattern)

        return jit(compute)
