from jaxtyping import Array, Shaped

# TODO: the dtypes are actually pretty well defined in the three main stages
# bayer, rgb and yuv and should be properly defined here

BayerMosaic = Shaped[Array, "h w"]
BayerChannels = Shaped[BayerMosaic, "4 h/2 w/2"]
RGBImage = Shaped[Array, "h w 3"]

Tensor2D = Shaped[Array, "a b"]
Tensor3D = Shaped[Array, "a b c"]
