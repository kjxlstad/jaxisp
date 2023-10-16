from functools import partial
from typing import Callable

from jax import jit
from jax.image import resize
from jaxtyping import Array, Shaped
from pydantic import validate_call


@validate_call
def scl[Input: Shaped[Array, "h w 3"], Output: Shaped[Array, "h w 3"]](
    width: int,
    height: int,
) -> Callable[[Input], Output]:
    # TODO: replace with cv2, orders of magnitude faster
    return jit(partial(
        resize, shape=(height, width, 3), method="linear")
    )
