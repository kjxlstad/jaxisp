from functools import partial

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped

from jaxisp.helpers import BayerPattern, mean_filter, merge_bayer, split_bayer
from jaxisp.nodes.common import ISPNode


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
        (jnp.abs(r - avg_r) > diff_threshold)
        * (jnp.abs(r - avg_g) > diff_threshold)
        * (jnp.abs(avg_r - avg_g) > diff_threshold)
        * (jnp.abs(avg_r - avg_b) < diff_threshold)
    )

    is_b_noise = (
        (jnp.abs(b - avg_g) > diff_threshold)
        * (jnp.abs(b - avg_r) > diff_threshold)
        * (jnp.abs(avg_b - avg_g) > diff_threshold)
        * (jnp.abs(avg_b - avg_r) < diff_threshold)
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

    weight = jnp.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.1])
    lower_bound = jnp.array([0, 30, 50, 70, 100, 150, 200])
    upper_bound = jnp.array([30, 50, 70, 100, 150, 200, 250])
    fade_1 = piecewise_weight(y, weight, lower_bound, upper_bound)
    fade_2 = piecewise_weight(avg_c1, weight, lower_bound, upper_bound)

    fade = fade_1 * fade_2
    return fade * chroma_corrected + (1 - fade) * array


class CNF(ISPNode):
    def compile(
        self,
        bayer_pattern: str,
        diff_threshold: int,
        r_gain: int,
        b_gain: int,
        saturation_hdr: int,
        **kwargs,
    ):
        bayer_pattern = BayerPattern[bayer_pattern.upper()]

        def compute(
            bayer_mosaic: Shaped[Array, "h w"]
        ) -> Shaped[Array, "h w"]:
            channels = split_bayer(bayer_mosaic, pattern=bayer_pattern)
            r, gr, gb, b = channels

            avg_r, avg_g, avg_b, is_r_noise, is_b_noise = compute_noise_diff(
                channels, diff_threshold
            )

            y = (306 * avg_r + 601 * avg_g + 117 * avg_b) >> 10
            r_cnc = compute_noise_correction(r, avg_g, avg_r, avg_b, y, r_gain)
            b_cnc = compute_noise_correction(b, avg_g, avg_b, avg_r, y, b_gain)
            r_cnc = is_r_noise * r_cnc + ~is_r_noise * r
            b_cnc = is_b_noise * b_cnc + ~is_b_noise * b

            bayer = merge_bayer(
                jnp.stack([r_cnc, gr, gb, b_cnc]), pattern=bayer_pattern
            )
            return jnp.clip(bayer, 0, saturation_hdr)

        return jit(compute)
