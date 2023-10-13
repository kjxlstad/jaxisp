from jaxtyping import Array, Shaped

# TODO: the dtypes are actually pretty well defined in the three main stages
# bayer, rgb and yuv and should be properly defined here

ImageYUV = Shaped[Array, "h w 3"]
ImageRGB = Shaped[Array, "h w 3"]

# TODO: consider defining separate in- and output names for bayer and channels,
# that way they are reusable e.g. make below prettier
# BayerImage = Shaped[Array, "h w"]
# BayerChannels = Shaped[Array, "4 h/2 w/2"]
# BayerDemosaic = Shaped[Array, "4 h w"]
# BayerMosaic = Shaped[Array, "h*2 w*2"]
