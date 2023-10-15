from itertools import product

import jax.numpy as jnp
from pydantic.dataclasses import dataclass

from jaxisp.helpers import mean_filter, neighbor_windows_slice, pad_spatial
from jaxisp.nodes.common import ISPNode
from jaxisp.type_utils import ImageYUV


@dataclass
class NLM(ISPNode):
    window_size: int
    patch_size: int
    h: int

    def compile(self):
        dist = jnp.arange(255**2)
        lut = (1024 * jnp.exp(-dist / self.h**2)).astype(jnp.int32)

        def compute(array: ImageYUV) -> ImageYUV:
            padded = pad_spatial(array, padding=self.window_size // 2)

            nlm_y_image = jnp.zeros_like(array)
            weights = jnp.zeros_like(array)

            for x, y in product(range(self.window_size), repeat=2):
                window = neighbor_windows_slice(padded, self.window_size, x, y)
                distance = mean_filter((array - window) ** 2, self.patch_size)

                weight = lut[distance]

                nlm_y_image += window * weight
                weights += weight

            return (nlm_y_image / weights).astype(jnp.uint8)

        return compute
