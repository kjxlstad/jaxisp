from functools import partial

from jax import jit
from jax.image import resize
from pydantic import validate_call


@validate_call
def scl(
    width: int,
    height: int,
):
    # TODO: replace with cv2, orders of magnitude faster
    return jit(partial(
        resize, shape=(height, width, 3), method="linear")
    )
