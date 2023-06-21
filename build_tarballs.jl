using BinaryBuilder, Pkg

name = "GALAHAD"
version = v"5.0.0"

# Collection of sources required to complete build
sources = [
    GitSource("https://github.com/ralna/GALAHAD.git", "513e4378a7cb3611cd5dbe1575fe6bc12218ee22"),
]

# Bash recipe for building across all platforms
script = raw"""
cd GALAHAD

if [[ "${target}" == *mingw* ]]; then
  MPI="msmpi"
  LBT="blastrampoline-5"
else
  MPI="mpifort"
  LBT="blastrampoline"
fi

meson setup builddir --cross-file=${MESON_TARGET_TOOLCHAIN} \
                     --buildtype=release \
                     --default-library=shared \
                     --prefix=$prefix \
                     -Dlibblas=$LBT \
                     -Dliblapack=$LBT \
                     -Dexamples=false \
                     -Dtests=false

meson compile -C builddir
meson install -C builddir
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = supported_platforms()
platforms = expand_gfortran_versions(platforms)

products = [
    LibraryProduct("libgalahad_single", :libgalahad_single),
    LibraryProduct("libgalahad_double", :libgalahad_double)
]

# Dependencies that must be installed before this package can be built
dependencies = [
    Dependency(PackageSpec(name="Hwloc_jll", uuid="e33a78d0-f292-5ffc-b300-72abe9b543c8")),
    Dependency(PackageSpec(name="MUMPS_seq_jll", uuid="d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d")),
    Dependency(PackageSpec(name="METIS_jll", uuid="d00139f3-1899-568f-a2f0-47f597d42d70")),
    Dependency(PackageSpec(name="libblastrampoline_jll", uuid="8e850b90-86db-534c-a0d3-1478176c7d93"), compat="5.4.0"),
    Dependency(PackageSpec(name="CompilerSupportLibraries_jll", uuid="e66e0078-7015-5450-92f7-15fbd957f2ae"))
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; julia_compat="v1.9")
