import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic import validate_call

from jaxisp.helpers import merge_bayer, split_bayer
from jaxisp.nodes.common import SensorConfig


@validate_call
def blc(
    alpha: int,
    beta: int,
    black_level_r: int,
    black_level_gr: int,
    black_level_gb: int,
    black_level_b: int,
    sensor: SensorConfig,
):
    def compute(
        bayer_mosaic: Shaped[Array, "h w"]
    ) -> Shaped[Array, "h w"]:
        r, gr, gb, b = split_bayer(bayer_mosaic, sensor.bayer_pattern)

        r = jnp.clip(r - black_level_r, 0)
        gr -= black_level_gr - jnp.right_shift(r * alpha, 10)
        gb -= black_level_gb - jnp.right_shift(b * beta, 10)
        b = jnp.clip(b - black_level_b, 0)

        return merge_bayer(
            jnp.stack([r, gr, gb, b]), sensor.bayer_pattern
        )

    return jit(compute)
