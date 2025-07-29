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

## Environment variables

Note that the following environment variables must be set before starting Julia for the default sparse linear solver `SSIDS`:
```raw
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
```

## LibHSL

We highly recommend to download [libHSL](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL) and install the official version of `HSL_jll.jl`.
This optional dependency provides access to more reliable and powerful linear solvers in `GALAHAD.jl`.
Note that this requires at least version `5.3.0` of `GALAHAD` and version `2025.7.21` of `libHSL`, both of which are provided via `GALAHAD_jll.jl` and `HSL_jll.jl`.

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
shared libraries `libgalahad_single`, `libgalahad_double` and `libgalahad_quadruple`
before `using GALAHAD`.

```bash
export JULIA_GALAHAD_LIBRARY_PATH=/home/alexis/Applications/GALAHAD/lib
```

The environment variable `JULIA_GALAHAD_LIBRARY_PATH` can be set
permanently in your shell's startup file (e.g., `.bashrc`)
or in Julia's startup file at `$HOME/.julia/config/startup.jl`.

You can also define it directly from within Julia:
```julia
ENV["JULIA_GALAHAD_LIBRARY_PATH"] = "/home/alexis/Applications/GALAHAD/lib"
```

You can check whether you're using the default precompiled libraries (`"YGGDRASIL"`)
or your own local ones (`"CUSTOM"`) by inspecting the constant `GALAHAD_INSTALLATION` in `GALAHAD.jl`:
```julia
using GALAHAD
GALAHAD.GALAHAD_INSTALLATION
```

If you have set the environment variable `JULIA_GALAHAD_LIBRARY_PATH` but `GALAHAD_INSTALLATION` still shows `"YGGDRASIL"`,
you may need to regenerate the cache of `GALAHAD.jl` by running:

```julia
force_recompile(package_name::String) = Base.compilecache(Base.identify_package(package_name))
force_recompile("GALAHAD")
```
