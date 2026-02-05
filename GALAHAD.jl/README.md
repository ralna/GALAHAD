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

## Documentation

Documentation is available online from [https://ralna.github.io/galahad_docs/html/Julia](https://ralna.github.io/galahad_docs/html/Julia).

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

## BLAS and LAPACK demuxer

`GALAHAD_jll.jl` is compiled with [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline) (LBT), a library that can change between BLAS and LAPACK backends at runtime such as OpenBLAS, Intel MKL, BLIS, and Apple Accelerate.
The default BLAS and LAPACK backend used in the Julia interface `GALAHAD.jl` is [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS).

### Display backends

Check what backends are loaded using:
```julia
import LinearAlgebra
LinearAlgebra.BLAS.lbt_get_config()
```

### Sequential BLAS and LAPACK

If you have both the LP64 and ILP64 [reference versions of BLAS and LAPACK](https://github.com/Reference-LAPACK/lapack) installed, you can switch to the sequential backends by running:
```julia
using ReferenceBLAS32_jll, ReferenceBLAS_jll, LAPACK32_jll, LAPACK_jll
LinearAlgebra.BLAS.lbt_forward(libblas32, clear=true)
LinearAlgebra.BLAS.lbt_forward(liblapack32)
LinearAlgebra.BLAS.lbt_forward(libblas, suffix_hint="64_")
LinearAlgebra.BLAS.lbt_forward(liblapack, suffix_hint="64_")
using GALAHAD
```

### MKL

If you have [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) installed,
switch to MKL by adding `using MKL` to your code:

```julia
using MKL
using GALAHAD
```

### AppleAccelerate

If you are using macOS ≥ v13.4 and you have [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl) installed, add `using AppleAccelerate` to your code:

```julia
using AppleAccelerate
using GALAHAD
```

## Local version of GALAHAD.jl

If you want to add and test new features in `GALAHAD.jl`, you can use a local version with the following commands
(if you are in the root folder of `GALAHAD.jl`):

```julia
using Pkg
Pkg.develop(path="./GALAHAD.jl")
Pkg.test("GALAHAD")
````

You can verify that the local version is being used if you see a path on the right of the version number with:

```julia
using Pkg
Pkg.status("GALAHAD")
```

If you install `GALAHAD.jl` normally with the following command, the local version will no longer be tracked, and the latest version registered in the Julia General Registry will be used instead:

```julia
using Pkg
Pkg.add("GALAHAD")
```

## Custom shared libraries

GALAHAD is already precompiled with [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) for all platforms.
The Julia package [GALAHAD_jll.jl](https://github.com/JuliaBinaryWrappers/GALAHAD_jll.jl)
is a dependency of `GALAHAD.jl` and handles the automatic download of a
precompiled version of GALAHAD for you.

To facilitate testing of new features by GALAHAD developers or enable
advanced users to utilize commercial linear solvers like `PARDISO` or
`WSMP`, it is also possible to bypass reliance on precompiled shared
libraries.
This is particularly relevant when new symbols required for upcoming GALAHAD
packages have not yet been included in an official release and are therefore
not available in the shared libraries provided by `GALAHAD_jll.jl`.

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
