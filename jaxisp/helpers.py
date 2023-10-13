from enum import Enum
from itertools import product
from functools import partial
from typing import Self

from jax import jit, vmap, lax
import jax.numpy as jnp

from jaxisp.array_types import BayerMosaic, BayerChannels, Tensor2D, Tensor3D


class BayerPattern(Enum):
    RGGB = 0, 1, 2, 3
    GBRG = 1, 2, 0, 3
    BGGR = 2, 3, 1, 0
    GRBG = 0, 2, 1, 3

    @classmethod
    def from_str(cls, string: str) -> Self:
        return cls[string.upper()]


@partial(jit, static_argnums=(1,))
def split_bayer(bayer_image: BayerMosaic, pattern: BayerPattern) -> BayerChannels:
    """Splits a zero channel bayer mosaic matrix into its 4 color-channels."""
    channels = [bayer_image[y::2, x::2] for y, x in product((0, 1), repeat=2)]
    return jnp.stack([channels[i] for i in pattern.value])


@partial(jit, static_argnums=(1,))
def merge_bayer(channels: BayerChannels, pattern: BayerPattern) -> BayerMosaic:
    _, h_half, w_half = channels.shape

    r0c0, r0c1, r1c0, r1c1 = (channels[i] for i in pattern.value)
    bayer_image = jnp.empty((2 * h_half, 2 * w_half), dtype=channels.dtype)

    return (
        bayer_image.at[0::2, 0::2].set(r0c0).at[0::2, 1::2].set(r0c1).at[1::2, 0::2].set(r1c0).at[1::2, 1::2].set(r1c1)
    )


# TODO: figure this out better
# This axis order is quite weird, this is just a reordered normal sliding window
# TODO: add good typing
# This works, TODO: should be type hinted that it supports spatial images, ex. [H, W] or
def lazy_neighbor_windows(array: Tensor2D, window_size: int = 3):
    assert window_size % 2 == 1, "Window size must be odd"
    height = array.shape[0] - window_size + 1
    width = array.shape[1] - window_size + 1

    for j in range(window_size):
        for i in range(window_size):
            yield array[j : j + height, i : i + width, ...]


# TODO: research speed of this compared to just calling list(lazy_neighbor_windows)
@partial(jit, static_argnums=(1,))
def neighbor_windows(array: Tensor2D, window_size: int = 3) -> Tensor3D:
    assert window_size % 2 == 1, "Window size must be odd"
    height = array.shape[0] - window_size + 1
    width = array.shape[1] - window_size + 1

    return jnp.stack([array[j : j + height, i : i + width, ...] for j, i in product(range(window_size), repeat=2)])


# TODO: deprecate, padded windows should be enough
@partial(jit, static_argnums=(1,))
def bayer_neighbor_pixels(array: Tensor2D, pattern: BayerChannels) -> Tensor3D:
    sliding_windows = vmap(partial(neighbor_windows, window_size=3), out_axes=1)
    padded = pad_spatial(array, padding=2, mode="reflect")
    channels = split_bayer(padded, pattern=pattern)
    return sliding_windows(channels)


@partial(jit, static_argnums=(1, 2))
def pad_spatial(array, padding: int, mode: str = "reflect"):
    # TODO: a bunch of headaches could be alleviated by using channel first axes order
    spatial_dims = (0, 1)
    padding = [(padding, padding) if dim in spatial_dims else (0, 0) for dim in range(array.ndim)]
    return jnp.pad(array, padding, mode=mode)


@partial(jit, static_argnums=(1,))
def mean_filter(array: Tensor2D, window_size: int) -> Tensor3D:
    assert window_size % 2 == 1, "Filter size must be odd"
    padded = pad_spatial(array, window_size // 2, mode="reflect")
    windows = lazy_neighbor_windows(padded, window_size=window_size)
    return (sum(windows) / window_size**2).astype(array.dtype)
