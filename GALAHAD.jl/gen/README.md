# Wrapping headers

This directory contains a script `wrapper.jl` that can be used to
automatically generate Julia wrappers from the C headers of GALAHAD.
This is done using [Clang.jl](https://github.com/JuliaInterop/Clang.jl).

# Usage

Activate and instantiate the project environment in this folder
to install the dependencies `Clang.jl` and `JuliaFormatter.jl`:
```julia
julia --project
julia> ]
(gen) pkg> instantiate
```

Then, regenerate the Julia wrappers with the following commands:
```julia
julia> include("wrapper.jl")
julia> main()
```

If you have already instantiated the environment, you can also run:
```bash
julia --project wrapper.jl
```

The `main` function supports the boolean keyword argument `optimized` to
clear the generated wrappers.  The default value of the argument
`optimized` is `true`.
You can also call `main(library)` if you want to
generate the wrapper for a specific GALAHAD `library`.

The possible values for `library` are:
- `"all"` (default);
- `"arc"`;
- `"bgo"`;
- `"blls"`;
- `"bllsb"`;
- `"bnls"`;
- `"bqp"`;
- `"bqpb"`;
- `"bsc"`;
- `"ccqp"`;
- `"cdqp"`;
- `"clls"`;
- `"convert"`;
- `"cqp"`;
- `"cro"`;
- `"demo"`;
- `"dgo"`;
- `"dlp"`;
- `"dps"`;
- `"dqp"`;
- `"eqp"`;
- `"expo"`;
- `"fdc"`;
- `"fdh"`;
- `"filtrane"`;
- `"fit"`;
- `"glrt"`;
- `"gls"`;
- `"gltr"`;
- `"hash"`;
- `"hsl"`;
- `"ir"`;
- `"l1qp"`;
- `"l2rt"`;
- `"lancelot"`;
- `"lhs"`;
- `"lls"`;
- `"llsr"`;
- `"llst"`;
- `"lms"`;
- `"lpa"`;
- `"lpb"`;
- `"lpqp"`;
- `"lqr"`;
- `"lpt"`;
- `"lsqp"`;
- `"lsrt"`;
- `"miqr"`;
- `"lstr"`;
- `"nls"`;
- `"nodend"`;
- `"presolve"`;
- `"psls"`;
- `"qp"`;
- `"qpa"`;
- `"qpb"`;
- `"qpc"`;
- `"roots"`;
- `"rpd"`;
- `"rqs"`;
- `"sbls"`;
- `"scu"`;
- `"sec"`;
- `"sha"`;
- `"sils"`;
- `"slls"`;
- `"sls"`;
- `"ssids`;
- `"ssls`;
- `"trb"`;
- `"trs"`;
- `"tru"`;
- `"ugo"`;
- `"uls"`;
- `"warm"`;
- `"wcp"`;
- `"ssids"`;
- `"hsl"`.

The Julia wrappers are generated in the directory `GALAHAD.jl/src/wrappers`.
If the `library` value is neither `hsl` nor `ssids`, we additionally create
a `Julia` folder within the package's directory (`GALAHAD/src/$library`) and
a symbolic link named `$library.jl`.
This link points to `GALAHAD/GALAHAD.jl/src/wrappers/$library.jl`.

# Maintenance

If a new package with a C interface is added, include an entry for it in
the `main` function of `wrapper.jl`, in the variable `packages` of
`rewriter.jl`, and in this `README.md`.  For instance, if the new
package is named `abcd`, insert the following line in `wrapper.jl`:

```julia
(name == "all" || name == "abcd") && wrapper("abcd", ["$galahad/galahad_abcd.h"], optimized, mp, run_sif=bool, run_qplib=bool)
```

If binaries are available for the new package, you also need to update the docstrings of `run_qplib` and `run_sif` in `GALAHAD/GALAHAD.jl/src/utils.jl` to add the symbol `:abcd` as an available solver.

Please also check the variables `nonparametric_structures_float` and
`nonparametric_structures_int` in `rewriter.jl` to specify whether a structure should be
parameterized to support various precisions (`Float32` / `Float64` / `Float128`)
or integer types (`Int32`, `Int64`).

The final step involves updating `GALAHAD.jl/src/GALAHAD.jl` by
appending the following two lines at the end of the file:

```julia
# abcd requires ...
include("wrappers/abcd.jl")
```

Now, the Julia wrappers for the `abcd` package are accessible upon
loading the Julia interface with `using GALAHAD`.

# Tests

The file `examples.jl` aids in generating Julia tests based on the
C tests within a GALAHAD package. To facilitate the translation of a C
test `GALAHAD/src/abcd/C/abcdtf.c` into a Julia test, follow these steps:

Add the following entry to `examples.jl`:

```julia
(name == "abcd") && examples("abcd", "tf")
```

Replace `"tf"` with `"t"` if translating a test from `abcdt.c`, or use `""` if no corresponding C test exists.

After including `examples.jl`, invoking `main("abcd")` will generate the file `test_abcd.jl` in the
directory `GALAHAD.jl/test`, along with a symbolic link pointing to this file in `GALAHAD/src/abcd/Julia`.

After manual modifications to ensure the Julia test works correctly, you can utilize `clean_example("abcd")`
to format the file.

To test the new package named `abcd` alongside other packages, insert the following line in `GALAHAD.jl/test/runtests.jl`:

```julia
include("test_abcd.jl")
```
