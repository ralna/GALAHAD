# Wrapping headers

This directory contains scripts that can be used to automatically generate wrappers for C headers by GALAHAD libraries.
This is done using Clang.jl.

# Usage

Either run `julia wrapper.jl` directly, or include it and call the `main()` function.
Be sure to activate the project environment in this folder, which will install `Clang.jl` and `JuliaFormatter.jl`.
The `main` function supports the boolean keyword argument `optimized` to clear the generated wrappers.
You can also call `main(library)` if you want to generate the wrapper for a specific GALAHAD `library`.
The possible values for `library` are:
- `"all"` (default);
- `"arc"`;
- `"bgo"`;
- `"blls"`;
- `"bqp"`;
- `"bqpb"`;
- `"bsc"`;
- `"ccqp"`;
- `"convert"`;
- `"cqp"`;
- `"cro"`;
- `"dgo"`;
- `"dps"`;
- `"dqp"`;
- `"eqp"`;
- `"fdc"`;
- `"fit"`;
- `"glrt"`;
- `"gls"`;
- `"gltr"`;
- `"hash"`;
- `"ir"`;
- `"l2rt"`;
- `"lhs"`;
- `"lms"`;
- `"lpa"`;
- `"lpb"`;
- `"lsqp"`;
- `"lsrt"`;
- `"lstr"`;
- `"nls"`;
- `"presolve"`;
- `"psls"`;
- `"qpa"`;
- `"qpb"`;
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
- `"trb"`;
- `"trs"`;
- `"tru"`;
- `"ugo"`;
- `"uls"`;
- `"wcp"`;
- `"ssids"`;
- `"hsl"`.
