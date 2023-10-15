"""Dead Pixel Correction
TODO: add link to diagram
"""
from enum import Enum

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped

from jaxisp.helpers import BayerPattern, bayer_neighbor_pixels, merge_bayer
from jaxisp.nodes.common import ISPNode

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


class DPCMode(Enum):
    GRADIENT = "gradient"
    MEAN = "mean"

    @classmethod
    def correction_func(cls, value: "DPCMode"):
        return {DPCMode.GRADIENT: gradient_dpc, DPCMode.MEAN: mean_dpc}[value]


def gradient_dpc(grid, diff_threshold):
    center = grid[C]
    neighbors = grid[[NW, N, NE, W, E, SW, S, SE], ...]

    neighbors_diff = jnp.abs(center - neighbors)
    mask = jnp.all(neighbors_diff > diff_threshold, axis=0)

    diff_stack = jnp.stack(
        [
            jnp.abs(2 * center - grid[N] - grid[S]),
            jnp.abs(2 * center - grid[W] - grid[E]),
            jnp.abs(2 * center - grid[NW] - grid[SE]),
            jnp.abs(2 * center - grid[SW] - grid[NE]),
        ],
        axis=-1,
    )

    indices = jnp.argmin(diff_stack, axis=-1, keepdims=True)

    neighbor_stack = jnp.stack(
        [
            grid[N] + grid[S],
            grid[W] + grid[E],
            grid[NW] + grid[SE],
            grid[SW] + grid[NE],
        ],
        axis=-1,
    )

    dpc_array = jnp.take_along_axis(
        neighbor_stack >> 1, indices, axis=-1
    ).squeeze(-1)
    return mask * dpc_array + ~mask * center


def mean_dpc(grid, diff_threshold):
    center = grid[C]
    neighbors = grid[[NW, N, NE, W, E, SW, S, SE], ...]

    neighbors_diff = jnp.abs(center - neighbors)
    mask = jnp.all(neighbors_diff > diff_threshold, axis=0)

    dpc_array = jnp.sum(grid[[N, W, E, S], ...], axis=0) // 4
    return mask * dpc_array + ~mask * center


class DPC(ISPNode):
    def compile(
        self,
        bayer_pattern: str,
        mode: DPCMode,
        diff_threshold: int,
        **kwargs,
    ):
        correction_func = DPCMode.correction_func(mode)
        bayer_pattern = BayerPattern[bayer_pattern.upper()]

        def compute(
            bayer_mosaic: Shaped[Array, "h w"]
        ) -> Shaped[Array, "h w"]:
            grid = bayer_neighbor_pixels(bayer_mosaic, pattern=bayer_pattern)
            res_array = correction_func(grid, diff_threshold)
            return merge_bayer(res_array, pattern=bayer_pattern)

        return jit(compute)
