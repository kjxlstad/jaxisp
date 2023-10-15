import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Shaped
from pydantic.dataclasses import dataclass

from jaxisp.helpers import merge_bayer, split_bayer
from jaxisp.nodes.common import ISPNode, SensorConfig


@dataclass
class BLC(ISPNode):
    alpha: int
    beta: int
    black_level_r: int
    black_level_gr: int
    black_level_gb: int
    black_level_b: int

    sensor: SensorConfig


    def compile(self):
        def compute(
            bayer_mosaic: Shaped[Array, "h w"]
        ) -> Shaped[Array, "h w"]:
            r, gr, gb, b = split_bayer(bayer_mosaic, self.sensor.bayer_pattern)

            r = jnp.clip(r - self.black_level_r, 0)
            gr -= self.black_level_gr - jnp.right_shift(r * self.alpha, 10)
            gb -= self.black_level_gb - jnp.right_shift(b * self.beta, 10)
            b = jnp.clip(b - self.black_level_b, 0)

            return merge_bayer(
                jnp.stack([r, gr, gb, b]), self.sensor.bayer_pattern
            )

        return jit(compute)
