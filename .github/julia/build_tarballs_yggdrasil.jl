# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

haskey(ENV, "GALAHAD_RELEASE") || error("The environment variable GALAHAD_RELEASE is not defined.")
haskey(ENV, "GALAHAD_COMMIT") || error("The environment variable GALAHAD_COMMIT is not defined.")
haskey(ENV, "GALAHAD_URL") || error("The environment variable GALAHAD_URL is not defined.")

name = "GALAHAD"
version = VersionNumber(ENV["GALAHAD_RELEASE"])

# Collection of sources required to complete build
sources = [
    GitSource(ENV["GALAHAD_URL"], ENV["GALAHAD_COMMIT"]),
]

# Bash recipe for building across all platforms
script = raw"""
# Update Ninja
cp ${host_prefix}/bin/ninja /usr/bin/ninja

cd ${WORKSPACE}/srcdir/GALAHAD

if [[ "${target}" == *mingw* ]]; then
  LBT="blastrampoline-5"
  HWLOC="hwloc-15"
else
  LBT="blastrampoline"
  HWLOC="hwloc"
fi

meson setup builddir_int32 --cross-file=${MESON_TARGET_TOOLCHAIN} \
                           --prefix=$prefix \
                           --buildtype=debug \
                           -Dint64=false \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=$LBT \
                           -Dliblapack=$LBT \
                           -Dlibsmumps=smumps \
                           -Dlibdmumps=dmumps \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=true \
                           -Dlibhsl=hsl_subset \
                           -Dlibhsl_modules=$prefix/modules

meson compile -C builddir_int32

meson setup builddir_int64 --cross-file=${MESON_TARGET_TOOLCHAIN} \
                           --prefix=$prefix \
                           --buildtype=debug \
                           -Dint64=true \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=$LBT \
                           -Dliblapack=$LBT \
                           -Dlibsmumps= \
                           -Dlibdmumps= \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=true \
                           -Dlibhsl=hsl_subset_64 \
                           -Dlibhsl_modules=$prefix/modules

meson compile -C builddir_int64

meson install -C builddir_int32
meson install -C builddir_int64
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = supported_platforms()

# The products that we will ensure are always built
products = [
    LibraryProduct("libgalahad_single", :libgalahad_single),
    LibraryProduct("libgalahad_double", :libgalahad_double),
    LibraryProduct("libgalahad_quadruple", :libgalahad_quadruple),
    LibraryProduct("libgalahad_single_64", :libgalahad_single_64),
    LibraryProduct("libgalahad_double_64", :libgalahad_double_64),
    LibraryProduct("libgalahad_quadruple_64", :libgalahad_quadruple_64),
]

# Dependencies that must be installed before this package can be built
dependencies = [
    HostBuildDependency(PackageSpec(name="Ninja_jll", uuid="76642167-d241-5cee-8c94-7a494e8cb7b7")),
    Dependency(PackageSpec(name="CompilerSupportLibraries_jll", uuid="e66e0078-7015-5450-92f7-15fbd957f2ae")),
    Dependency(PackageSpec(name="LLVMOpenMP_jll", uuid="1d63c593-3942-5779-bab2-d838dc0a180e"); platforms=filter(Sys.isbsd, platforms)),
    Dependency(PackageSpec(name="libblastrampoline_jll", uuid="8e850b90-86db-534c-a0d3-1478176c7d93"), compat="5.4.0"),
    Dependency(PackageSpec(name="Hwloc_jll", uuid="e33a78d0-f292-5ffc-b300-72abe9b543c8")),
    Dependency(PackageSpec(name="MUMPS_seq_jll", uuid="d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d")),
    Dependency(PackageSpec(name="HSL_jll", uuid="017b0a0e-03f4-516a-9b91-836bbd1904dd")),
    # Dependency(PackageSpec(name="PaStiX_jll", uuid="46e5285b-ff06-5712-adf2-cc145d39f096")),
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; preferred_gcc_version=v"9.1.0", julia_compat="1.9", clang_use_lld=false)
