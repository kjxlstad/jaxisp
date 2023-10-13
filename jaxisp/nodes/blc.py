from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Shaped

from jaxisp.nodes.common import ISPNode
from jaxisp.helpers import BayerPattern, split_bayer, merge_bayer


class BLC(ISPNode):
    def compile(
        self, bayer_pattern: str, alpha: int, beta: int, bl_r: int, bl_gr: int, bl_gb: int, bl_b: int, **kwargs
    ):
        bayer_pattern = BayerPattern[bayer_pattern.upper()]

        def compute(bayer_mosaic: Shaped[Array, "h w"]) -> Shaped[Array, "h w"]:
            r, gr, gb, b = split_bayer(bayer_mosaic, pattern=bayer_pattern)

            r = jnp.clip(r - bl_r, 0)
            gr -= bl_gr - jnp.right_shift(r * alpha, 10)
            gb -= bl_gb - jnp.right_shift(b * beta, 10)
            b = jnp.clip(b - bl_b, 0)

            return merge_bayer(jnp.stack([r, gr, gb, b]), pattern=bayer_pattern)

        return jit(compute)
