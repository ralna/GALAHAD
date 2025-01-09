using BinaryBuilder, Pkg

haskey(ENV, "GALAHAD_RELEASE") || error("The environment variable GALAHAD_RELEASE is not defined.")
haskey(ENV, "GALAHAD_COMMIT") || error("The environment variable GALAHAD_COMMIT is not defined.")

name = "GALAHAD"
version = VersionNumber(ENV["GALAHAD_RELEASE"])

# Collection of sources required to complete build
sources = [
    GitSource("https://github.com/ralna/GALAHAD.git", ENV["GALAHAD_COMMIT"])
]

# Bash recipe for building across all platforms
script = raw"""
# Export dependencies
mkdir ${prefix}/deps
cd ${libdir}
for file in $(ls .); do
   if [[ -f $file ]]; then
      if [[ -z $(ls -la $file | grep 'artifacts') ]]; then
         cp -P ${file} ${prefix}/deps/${file}
      else
         cp -L ${file} ${prefix}/deps/${file}
      fi
   fi
done
cd ${prefix}
cp -rL share/licenses deps/licenses
chmod -R u=rwx deps
tar -czvf deps.tar.gz deps
rm -r deps

# Update Ninja
cp ${host_prefix}/bin/ninja /usr/bin/ninja

cd ${WORKSPACE}/srcdir/GALAHAD

if [[ "${target}" == *mingw* ]]; then
  HWLOC="hwloc-15"
else
  HWLOC="hwloc"
fi

meson setup builddir_int32 --cross-file=${MESON_TARGET_TOOLCHAIN%.*}_gcc.meson \
                           --prefix=$prefix \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=openblas \
                           -Dliblapack=openblas \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=true \
                           -Dint64=false \
                           -Dlibhsl= \
                           -Dlibhsl_modules=$prefix/modules

meson compile -C builddir_int32
meson install -C builddir_int32

meson setup builddir_int64 --cross-file=${MESON_TARGET_TOOLCHAIN%.*}_gcc.meson \
                           --prefix=$prefix \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=openblas64_ \
                           -Dliblapack=openblas64_ \
                           -Dlibsmumps= \
                           -Dlibdmumps= \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=true \
                           -Dint64=true \
                           -Dlibhsl= \
                           -Dlibhsl_modules=$prefix/modules

meson compile -C builddir_int64
meson install -C builddir_int64
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = supported_platforms()
platforms = expand_gfortran_versions(platforms)

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
    Dependency(PackageSpec(name="OpenBLAS32_jll", uuid="656ef2d0-ae68-5445-9ca0-591084a874a2")),
    Dependency(PackageSpec(name="OpenBLAS_jll", uuid="4536629a-c528-5b80-bd46-f80d51c5b363")),
    Dependency(PackageSpec(name="Hwloc_jll", uuid="e33a78d0-f292-5ffc-b300-72abe9b543c8")),
    Dependency(PackageSpec(name="MUMPS_seq_jll", uuid="d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"), compat="=5.4.1"),
    Dependency(PackageSpec(name="HSL_jll", uuid="017b0a0e-03f4-516a-9b91-836bbd1904dd")),
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; preferred_gcc_version=v"9.1.0", julia_compat="1.6")
