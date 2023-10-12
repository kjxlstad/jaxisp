"""Dead Pixel Correction
TODO: add link to diagram
"""

from jax import jit
import jax.numpy as jnp

from jaxisp.nodes.common import ISPNode
from jaxisp.helpers import BayerPattern, bayer_neighbor_pixels, merge_bayer
from jaxisp.array_types import BayerMosaic

# shorthand cardinal directions
NW = 0
N = 1
NE = 2
W = 3
C = 4
E = 5
SW = 6
S = 7
SE = 8


class DPC(ISPNode):
    def compile(
        self,
        bayer_pattern: str,
        diff_threshold: int,
        **kwargs,
    ):
        bayer_pattern = BayerPattern[bayer_pattern.upper()]

        def compute(array: BayerMosaic) -> BayerMosaic:
            grid = bayer_neighbor_pixels(array, pattern=bayer_pattern)

            center = grid[C]
            neighbors = grid[[NW, N, NE, W, E, SW, S, SE], ...]

            neighbors_diff = jnp.abs(center - neighbors)
            mask = jnp.all(neighbors_diff > diff_threshold, axis=0)

            diff_stack = jnp.stack([
                jnp.abs(2 * center - grid[N] - grid[S]),
                jnp.abs(2 * center - grid[W] - grid[E]),
                jnp.abs(2 * center - grid[NW] - grid[SE]),
                jnp.abs(2 * center - grid[SW] - grid[NE]),
            ], axis=-1)

            indices = jnp.argmin(diff_stack, axis=-1, keepdims=True)

            neighbor_stack = jnp.stack([
                grid[N] + grid[S],
                grid[W] + grid[E],
                grid[NW] + grid[SE],
                grid[SW] + grid[NE],
            ], axis=-1)

            dpc_array = jnp.take_along_axis(neighbor_stack >> 1, indices, axis=-1).squeeze(-1)
            res_array = mask * dpc_array + ~mask * center

            return merge_bayer(res_array, pattern=bayer_pattern)

        return jit(compute)
