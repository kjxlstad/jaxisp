# (WIP) jaxisp
Jax reimplementation of [openISP](https://github.com/cruxopen/openISP) heavily based on [fast-OpenISP](https://github.com/QiuJueqin/fast-openISP).

Speeds up openISP by being jit-compiled wherever possible and easily enabling hardware-agnostic parallelization, while being compile-time type-safe using [beartype](https://github.com/beartype/beartype) and [jaxtyping](https://github.com/google/jaxtyping). Thorougly property-tested using [hypothesis](https://github.com/HypothesisWorks/hypothesis).
