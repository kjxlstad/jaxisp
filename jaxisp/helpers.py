from enum import Enum
from itertools import product
from functools import partial

from jax import jit, vmap
from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Shaped


class BayerPattern(Enum):
    RGGB = 0, 1, 2, 3
    GBRG = 1, 2, 0, 3
    BGGR = 2, 3, 1, 0
    GRBG = 0, 2, 1, 3

    @classmethod
    def from_str(cls, string: str):
        return cls[string.upper()]


BayerMosaic = Shaped[Array, "h w"]
BayerChannels = Shaped[BayerMosaic, "4 h/2 w/2"]


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
@partial(jit, static_argnums=(1,))
def neighbor_windows(array, window_size: int = 3):
    assert window_size % 2 == 1
    height = array.shape[0] - window_size + 1
    width = array.shape[1] - window_size + 1

    return jnp.stack([array[..., j : j + height, i : i + width] for j, i in product(range(window_size), repeat=2)])


@partial(jit, static_argnums=(1,))
def bayer_neighbor_pixels(array, pattern: BayerChannels):
    sliding_windows = vmap(partial(neighbor_windows, window_size=3), in_axes=0, out_axes=1)
    return sliding_windows(split_bayer(jnp.pad(array, 2, mode="reflect"), pattern))
