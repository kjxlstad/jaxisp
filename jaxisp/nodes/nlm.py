from functools import partial

from jax import jit, vmap
import jax.numpy as jnp

from jaxisp.helpers import mean_filter, neighbor_windows, pad_spatial
from jaxisp.nodes.common import ISPNode
from jaxisp.array_types import ImageYUV


class NLM(ISPNode):
    def compile(self, window_size: int, patch_size: int, h: int, **kwargs):
        distance = jnp.arange(255**2)
        lut = (1024 * jnp.exp(-distance / h**2)).astype(jnp.int32)

        batched_mean_filter = jit(vmap(partial(mean_filter, window_size=patch_size)))

        def compute(array: ImageYUV) -> ImageYUV:
            padded = pad_spatial(array, padding=window_size // 2)
            windows = neighbor_windows(padded, window_size=window_size)

            distance = batched_mean_filter((array - windows) ** 2)
            weight = lut[distance]
            nlm_image = jnp.sum(windows * weight, axis=0) / jnp.sum(weight, axis=0)
            return nlm_image.astype(jnp.uint8)

        return jit(compute)
