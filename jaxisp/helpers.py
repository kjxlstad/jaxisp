from enum import Enum
from functools import partial, reduce
from itertools import product
from operator import add

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Shaped

from jaxisp.type_utils import WindowdFunc


class BayerPattern(Enum):
    RGGB = 0, 1, 2, 3
    GBRG = 1, 3, 0, 2
    BGGR = 3, 2, 1, 0
    GRBG = 2, 0, 3, 1


class CFAMode(Enum):
    BILINEAR = "bilinear"
    MALVAR = "malvar"


@partial(jit, static_argnums=(1,))
def split_bayer(
    bayer_image: Shaped[Array, "h w"], pattern: BayerPattern
) -> Shaped[Array, "4 h/2 w/2"]:
    """Splits a zero channel bayer mosaic matrix into its 4 color-channels."""
    channels = [bayer_image[y::2, x::2] for y, x in product((0, 1), repeat=2)]
    return jnp.stack([channels[i] for i in pattern.value])


@partial(jit, static_argnums=(1,))
def merge_bayer(
    channels: Shaped[Array, "4 h w"], pattern: BayerPattern
) -> Shaped[Array, "h*2 w*2"]:
    _, h_half, w_half = channels.shape

    r0c0, r0c1, r1c0, r1c1 = (channels[i] for i in pattern.value)
    bayer_image = jnp.empty((2 * h_half, 2 * w_half), dtype=channels.dtype)

    return (
        bayer_image
        .at[0::2, 0::2].set(r0c0)
        .at[0::2, 1::2].set(r0c1)
        .at[1::2, 0::2].set(r1c0)
        .at[1::2, 1::2].set(r1c1)
    )


@partial(jit, static_argnums=(1, 2, 3))
def neighbor_windows_slice(array, window_size: int, i: int, j: int):
    height = array.shape[0] - window_size + 1
    width = array.shape[0] - window_size + 1
    return array[j : j + height, i : i + width, ...]


@partial(jit, static_argnums=(0, 2))
def reduce_windows(
    fun: WindowdFunc, array: Shaped[Array, "h w ..."], window_size: int
) -> Shaped[Array, "h w ..."]:
    assert window_size % 2 == 1, "Window size must be odd"
    window_iterator = (
        neighbor_windows_slice(array, window_size, i, j)
        for i, j in product(range(window_size), repeat=2)
    )
    return reduce(fun, window_iterator)


@partial(jit, static_argnums=(1,))
def neighbor_windows(
    array: Shaped[Array, "h w ..."], window_size: int = 3
) -> Shaped[Array, "b _ _ ..."]:
    assert window_size % 2 == 1, "Window size must be odd"
    height = array.shape[0] - window_size + 1
    width = array.shape[1] - window_size + 1

    return jnp.stack(
        [
            array[j : j + height, i : i + width, ...]
            for j, i in product(range(window_size), repeat=2)
        ]
    )


@partial(jit, static_argnums=(1,))
def bayer_neighbor_pixels(
    array: Shaped[Array, "h w"], pattern: BayerPattern
) -> Shaped[Array, "9 4 h/2 w/2"]:
    sliding_windows = vmap(
        partial(neighbor_windows, window_size=3), out_axes=1
    )
    padded = pad_spatial(array, padding=2, mode="reflect")
    channels = split_bayer(padded, pattern=pattern)
    return sliding_windows(channels)


@partial(jit, static_argnums=(1, 2))
def pad_spatial(
    array: Shaped[Array, "h w ..."], padding: int, mode: str = "reflect"
) -> Shaped[Array, "_ _ ..."]:
    # TODO: a bunch of headaches could be alleviated
    # by using channel first axes order
    spatial_dims = (0, 1)
    padding = [
        (padding, padding) if dim in spatial_dims else (0, 0)
        for dim in range(array.ndim)
    ]
    return jnp.pad(array, padding, mode=mode)


# TODO: should be type hintable with typevar dtype
@partial(jit, static_argnums=(1,))
def mean_filter(
    array: Shaped[Array, "h w ..."], window_size: int
) -> Shaped[Array, "h w ..."]:
    assert window_size % 2 == 1, "Filter size must be odd"
    padded = pad_spatial(array, window_size // 2, mode="reflect")
    total = reduce_windows(add, padded, window_size)
    return (total / window_size**2).astype(array.dtype)

@partial(jit, static_argnums=(1,))
def gaussian_kernel(sigma: float, kernel_size: int):
    x = jnp.arange(kernel_size) - kernel_size // 2
    x += (0.5 if kernel_size % 2 == 0 else 0)
    y, x = jnp.meshgrid(x, x)
    kernel = jnp.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

@jit
def bilateral_filter(
    array,
    spatial_weight,
    intensity_weight_lut,
    right_shift=0
):
    filter_height, filter_width = spatial_weight.shape[:2]
    spatial_weight = spatial_weight.flatten()

    padded_array = pad_spatial(
        array, padding=filter_height // 2, mode="reflect")

    bf_array = jnp.zeros_like(array)
    weights = jnp.zeros_like(array)

    for i, (x, y) in enumerate(product(range(filter_width), repeat=2)):
        neighbor = neighbor_windows_slice(padded_array, filter_width, x, y)
        intensity_diff = (neighbor - array) ** 2
        weight = intensity_weight_lut[intensity_diff] * spatial_weight[i]
        weight = jnp.right_shift(weight, right_shift)

        bf_array += neighbor * weight
        weights += weight

    return (bf_array / weights).astype(array.dtype)
