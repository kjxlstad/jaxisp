from functools import partial

from jax import jit
from jax.image import resize

from jaxisp.nodes.common import ISPNode


class AWB(ISPNode):
    def compile(self, width: int, height: int, **kwargs):
        return jit(partial(resize, shape=(height, width, 3), method="linear"))
