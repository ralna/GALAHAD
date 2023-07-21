# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.

# julia --color=yes build_tarballs.jl x86_64-linux-gnu-libgfortran5,x86_64-apple-darwin-libgfortran4,aarch64-apple-darwin-libgfortran5,x86_64-w64-mingw32-libgfortran5 --debug --verbose --deploy="amontoison/GALAHAD_jll.jl"

# Supported Platforms:
# - aarch64-apple-darwin
# - aarch64-linux-gnu
# - aarch64-linux-musl
# - armv6l-linux-gnueabihf
# - armv6l-linux-musleabihf
# - armv7l-linux-gnueabihf
# - armv7l-linux-musleabihf
# - i686-linux-gnu
# - i686-linux-musl
# - i686-w64-mingw32
# - powerpc64le-linux-gnu
# - x86_64-apple-darwin
# - x86_64-linux-gnu
# - x86_64-linux-musl
# - x86_64-unknown-freebsd
# - x86_64-w64-mingw32

using BinaryBuilder, Pkg

name = "GALAHAD"
version = v"5.0.0"

# Collection of sources required to complete build
sources = [
    GitSource("https://github.com/ralna/GALAHAD.git", "ddff83fd43d84447b92b48d8857d50b731dd7fd8")
]

# Bash recipe for building across all platforms
script = raw"""
cd $WORKSPACE/srcdir/GALAHAD
cp include/* $includedir/
meson setup $libdir --cross-file=${MESON_TARGET_TOOLCHAIN} --buildtype=release -Dciface=true -Dlibblas=$libdir,libopenblas -Dliblapack=$libdir,libopenblas
meson compile -C $libdir
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = supported_platforms()
platforms = expand_gfortran_versions(platforms)

# The products that we will ensure are always built
products = [
    LibraryProduct("libgalahad", :libgalahad),
    LibraryProduct("libgalahad_hsl", :libgalahad_hsl),
    LibraryProduct("libgalahad_c", :libgalahad_c)
]

# Dependencies that must be installed before this package can be built
dependencies = [
    Dependency(PackageSpec(name="OpenBLAS32_jll", uuid="656ef2d0-ae68-5445-9ca0-591084a874a2")),
    Dependency(PackageSpec(name="CompilerSupportLibraries_jll", uuid="e66e0078-7015-5450-92f7-15fbd957f2ae"))
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; julia_compat="1.6", preferred_gcc_version = v"9.1.0")
