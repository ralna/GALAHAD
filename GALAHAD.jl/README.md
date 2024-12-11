# A [Julia](http://julialang.org) interface to [GALAHAD](https://www.galahad.rl.ac.uk/)

## Installation

`GALAHAD.jl` can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add GALAHAD
pkg> test GALAHAD
```

If you launch Julia from within the folder `GALAHAD.jl`, you can
directly run `pkg> dev .`.

## libHSL

We highly recommend to download [libHSL](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL) and install the official version of `HSL_jll.jl`.
This optional dependency gives you access to more reliable and powerful linear solvers.

## Documentation

Documentation is available online from [https://ralna.github.io/galahad_docs/html/Julia](https://ralna.github.io/galahad_docs/html/Julia).

## Custom shared libraries

GALAHAD is already precompiled with [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) for all platforms.
The Julia package [GALAHAD_jll.jl](https://github.com/JuliaBinaryWrappers/GALAHAD_jll.jl)
is a dependency of GALAHAD.jl and handles the automatic download of a
precompiled version of GALAHAD for you.

To facilitate testing of new features by GALAHAD developers or enable
advanced users to utilize commercial linear solvers like `PARDISO` or
`WSMP`, it is also possible to bypass reliance on precompiled shared
libraries.

To use your own installation of `GALAHAD`, set the environment variable
`JULIA_GALAHAD_LIBRARY_PATH` to point to the folder that contains the
shared libraries `libgalahad_single` and `libgalahad_double` before
`using GALAHAD`.

```bash
export JULIA_GALAHAD_LIBRARY_PATH=/home/alexis/Applications/GALAHAD/lib
```

The `JULIA_GALAHAD_LIBRARY_PATH` environment variable may be set
permanently in the shell's startup file, or in
`$HOME/.julia/config/startup.jl`.

