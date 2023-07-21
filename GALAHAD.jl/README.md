# A [Julia](http://julialang.org) Interface to [GALAHAD](https://www.galahad.rl.ac.uk/)

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** |
|:-----------------:|:-------------------------------:|:------------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://amontoison.github.io/GALAHAD.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://amontoison.github.io/GALAHAD.jl/dev
[build-gh-img]: https://github.com/amontoison/GALAHAD.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/amontoison/GALAHAD.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/amontoison/GALAHAD.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/amontoison/GALAHAD.jl
[codecov-img]: https://codecov.io/gh/amontoison/GALAHAD.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/amontoison/GALAHAD.jl

## Custom Installation

To use your custom GALAHAD, set the environment variable `JULIA_GALAHAD_LIBRARY_PATH` to point to the folder that contains shared libraries before `using GALAHAD`.

```bash
export JULIA_GALAHAD_LIBRARY_PATH=/home/alexis/Applications/GALAHAD/lib
```

The `JULIA_GALAHAD_LIBRARY_PATH` environment variable may be set permanently in the shell's startup file, or in `$HOME/.julia/config/startup.jl`.
