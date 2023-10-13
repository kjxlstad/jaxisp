# (WIP) jaxisp
Jax reimplementation of [openISP](https://github.com/cruxopen/openISP) heavily based on [fast-OpenISP](https://github.com/QiuJueqin/fast-openISP).

Speeds up openISP by being jit-compiled wherever possible and easily enabling hardware-agnostic parallelization, while being compile-time type-safe using [beartype](https://github.com/beartype/beartype) and [jaxtyping](https://github.com/google/jaxtyping). Thorougly property-tested using [hypothesis](https://github.com/HypothesisWorks/hypothesis).

| Node  | shape       | fast-openISP | jaxisp   |
|-------|-------------|--------------|----------|
| DPC   | 1000x1000   | 58.31 ms     | 25.24 ms |
| BLC   | 1000x1000   | 6.723 ms     | 3.387 ms |
| AAF   | 1000x1000   | 12.06 ms     | 12.79 ms |
| AWB   | 1000x1000   | 6.125 ms     | 3.352 ms |
| CNF   | 1000x1000   | 31.04 ms     | 3.359 ms |
| CFA   |             |              |          |
| CCM   | 1000x1000x3 | 17.14 ms     | 10.85 ms |
| GAC   | 1000x1000x3 | 13.56 ms     | 0.493 ms |
| CSC   |             |              |          |
| NLM   | 1000x1000x3 | 6321 ms      | 1968 ms  |
| BNF   |             |              |          |
| CEH   |             |              |          |
| EEH   |             |              |          |
| FCS   |             |              |          |
| HSC   |             |              |          |
| BBC   |             |              |          |
| Total |             | 6.47 s       | 2.02 s   |