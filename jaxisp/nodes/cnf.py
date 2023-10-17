from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call

from jaxisp.helpers import mean_filter, merge_bayer, split_bayer
from jaxisp.nodes.common import SensorConfig


# TODO: add output type
@partial(jit, static_argnums=(1,))
def compute_noise_diff(channels: Shaped[Array, "4 h w"], diff_threshold: int):
    r, gr, gb, b = channels
    avg_r = mean_filter(r, window_size=5)
    avg_g = (
        mean_filter(gr, window_size=5) + mean_filter(gb, window_size=5)
    ) >> 1
    avg_b = mean_filter(b, window_size=5)

    is_r_noise = (
        (r - avg_g > diff_threshold)
        * (r - avg_b > diff_threshold)
        * (avg_r - avg_g > diff_threshold)
        * (avg_r - avg_b < diff_threshold)
    )

    is_b_noise = (
        (b - avg_g > diff_threshold)
        * (b - avg_r > diff_threshold)
        * (avg_b - avg_g > diff_threshold)
        * (avg_b - avg_r < diff_threshold)
    )

    return avg_r, avg_g, avg_b, is_r_noise, is_b_noise


# TODO: type hint
@jit
def piecewise_weight(y, weight, lower_bound, upper_bound):
    predicate = (lower_bound.reshape(-1, 1, 1) < y) & (
        y <= upper_bound.reshape(-1, 1, 1)
    )
    return jnp.sum(predicate * weight.reshape(-1, 1, 1), axis=0)


# todo: type hint
@partial(jit, static_argnums=(5,))
def compute_noise_correction(array, avg_g, avg_c1, avg_c2, y, gain):
    damp_factor = jnp.select(
        [gain <= 1024, 1024 < gain <= 1229, 1229 < gain], [256, 128, 77]
    )

    max_avg = jnp.maximum(avg_g, avg_c2)
    signal_gap = array - max_avg
    chroma_corrected = max_avg + jnp.right_shift(damp_factor * signal_gap, 8)

    lower_bound = jnp.array([0, 30, 50, 70, 100, 150, 200])
    upper_bound = jnp.array([30, 50, 70, 100, 150, 200, 250])

    fade_1 = piecewise_weight(
        y,
        jnp.array([1., .9, .8, .7, .6, .3, .1]),
        lower_bound, upper_bound
    )

    fade_2 = piecewise_weight(
        y,
        jnp.array([1., .9, .8, .6, .5, .3]),
        lower_bound[:-1], upper_bound[:-1]
    )

    fade = fade_1 * fade_2
    return (fade * chroma_corrected + (1 - fade) * array).astype(jnp.int32)


@validate_call
def cnf[Input: Shaped[Array, "h w"], Output: Shaped[Array, "h w"]](
    diff_threshold: int,
    gain_r: int,
    gain_b: int,
    sensor: SensorConfig,
    saturation_hdr: int, # TODO: fixme
) -> Callable[[Input], Output]:
    def compute(bayer_mosaic: Input) -> Output:
        channels = split_bayer(bayer_mosaic, sensor.bayer_pattern)
        r, gr, gb, b = channels

        avg_r, avg_g, avg_b, is_r_noise, is_b_noise = compute_noise_diff(
            channels, diff_threshold
        )

        y = (306 * avg_r + 601 * avg_g + 117 * avg_b) >> 10
        r_cnc = compute_noise_correction(
            r, avg_g, avg_r, avg_b, y, gain_r)
        b_cnc = compute_noise_correction(
            b, avg_g, avg_b, avg_r, y, gain_b)
        r_cnc = is_r_noise * r_cnc + ~is_r_noise * r
        b_cnc = is_b_noise * b_cnc + ~is_b_noise * b

        bayer = merge_bayer(
            jnp.stack([r_cnc, gr, gb, b_cnc]),
            sensor.bayer_pattern
        )
        return jnp.clip(bayer, 0, saturation_hdr).astype(jnp.uint16)

    return jit(compute)
