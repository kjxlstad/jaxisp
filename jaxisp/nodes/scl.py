from functools import partial

from jax import jit
from jax.image import resize
from pydantic.dataclasses import dataclass

from jaxisp.nodes.common import ISPNode


@dataclass
class SCL(ISPNode):
    width: int
    height: int

    def compile(self):
        # TODO: replace with cv2, orders of magnitude faster
        return jit(partial(
            resize, shape=(self.height, self.width, 3), method="linear")
        )
