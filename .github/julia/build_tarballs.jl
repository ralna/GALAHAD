using BinaryBuilder, Pkg

haskey(ENV, "GALAHAD_RELEASE") || error("The environment variable GALAHAD_RELEASE is not defined.")
haskey(ENV, "GALAHAD_COMMIT") || error("The environment variable GALAHAD_COMMIT is not defined.")
haskey(ENV, "GALAHAD_URL") || error("The environment variable GALAHAD_URL is not defined.")

name = "GALAHAD"
version = VersionNumber(ENV["GALAHAD_RELEASE"])

# Collection of sources required to complete build
sources = [
    GitSource(ENV["GALAHAD_URL"], ENV["GALAHAD_COMMIT"]),
    ArchiveSource("https://mumps-solver.org/MUMPS_5.8.2.tar.gz", "eb515aa688e6dbab414bb6e889ff4c8b23f1691a843c68da5230a33ac4db7039")
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

# Compile MUMPS
cd $WORKSPACE/srcdir/MUMPS*

makefile="Makefile.G95.SEQ"
cp Make.inc/${makefile} Makefile.inc

# Add `-fallow-argument-mismatch` if supported
: >empty.f
FFLAGS=()
if gfortran -c -fallow-argument-mismatch empty.f >/dev/null 2>&1; then
    FFLAGS+=("-fallow-argument-mismatch")
fi
rm -f empty.*

if [[ "${target}" == *apple* ]]; then
    SONAME="-install_name"
else
    SONAME="-soname"
fi

BLAS_LAPACK="-L${libdir} -lopenblas"

make_args+=(OPTF="-O3"
            OPTL="-O3"
            OPTC="-O3"
            CDEFS=-DAdd_
            LMETISDIR=${libdir}
            IMETIS=-I${includedir}
            LMETIS="-L${libdir} -lmetis"
            ORDERINGSF="-Dpord -Dmetis"
            LIBEXT_SHARED=".${dlext}"
            SHARED_OPT="-shared"
            SONAME="${SONAME}"
            CC="$CC ${CFLAGS[@]}"
            FC="gfortran ${FFLAGS[@]}"
            FL="gfortran"
            RANLIB="echo"
            LIBBLAS="${BLAS_LAPACK}"
            LAPACK="${BLAS_LAPACK}")

make -j${nproc} allshared "${make_args[@]}"

mkdir ${includedir}/libseq
cp include/*.h ${includedir}
cp libseq/*.h ${includedir}/libseq
cp lib/*.${dlext} ${libdir}

# Compile GALAHAD
cd ${WORKSPACE}/srcdir/GALAHAD

if [[ "${target}" == *mingw* ]]; then
  HWLOC="hwloc-15"
else
  HWLOC="hwloc"
fi

QUADRUPLE="true"
if [[ "${target}" == *arm* ]] || [[ "${target}" == *aarch64-linux* ]] || [[ "${target}" == *aarch64-unknown-freebsd* ]] || [[ "${target}" == *powerpc64le-linux-gnu* ]] || [[ "${target}" == *riscv64* ]]; then
    QUADRUPLE="false"
fi

meson setup builddir_int32 --cross-file=${MESON_TARGET_TOOLCHAIN%.*}_gcc.meson \
                           --prefix=$prefix \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=openblas \
                           -Dliblapack=openblas \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=$QUADRUPLE \
                           -Dint64=false \
                           -Dexamples=false \
                           -Dtests=false \
                           -Dbinaries=true \
                           -Dlibhsl=hsl_subset \
                           -Dlibhsl_modules=$prefix/modules \
                           -Dlibcutest_modules=$prefix/modules

meson compile -C builddir_int32
meson install -C builddir_int32

meson setup builddir_int64 --cross-file=${MESON_TARGET_TOOLCHAIN%.*}_gcc.meson \
                           --prefix=$prefix \
                           -Dlibhwloc=$HWLOC \
                           -Dlibblas=openblas64_ \
                           -Dliblapack=openblas64_ \
                           -Dlibsmumps= \
                           -Dlibdmumps= \
                           -Dlibcutest_single= \
                           -Dlibcutest_double= \
                           -Dlibcutest_quadruple= \
                           -Dsingle=true \
                           -Ddouble=true \
                           -Dquadruple=$QUADRUPLE \
                           -Dint64=true \
                           -Dexamples=false \
                           -Dtests=false \
                           -Dbinaries=false \
                           -Dlibhsl=hsl_subset_64 \
                           -Dlibhsl_modules=$prefix/modules \
                           -Dlibcutest_modules=$prefix/modules

meson compile -C builddir_int64
meson install -C builddir_int64
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = supported_platforms()
platforms = expand_gfortran_versions(platforms)

# The products that we will ensure are always built
products = [
    LibraryProduct("libsmumps", :libsmumps),
    LibraryProduct("libdmumps", :libdmumps),
    LibraryProduct("libgalahad_single", :libgalahad_single),
    LibraryProduct("libgalahad_double", :libgalahad_double),
    # LibraryProduct("libgalahad_quadruple", :libgalahad_quadruple),
    LibraryProduct("libgalahad_single_64", :libgalahad_single_64),
    LibraryProduct("libgalahad_double_64", :libgalahad_double_64),
    # LibraryProduct("libgalahad_quadruple_64", :libgalahad_quadruple_64),
]

# Dependencies that must be installed before this package can be built
dependencies = [
    HostBuildDependency(PackageSpec(name="Ninja_jll", uuid="76642167-d241-5cee-8c94-7a494e8cb7b7")),
    Dependency(PackageSpec(name="CompilerSupportLibraries_jll", uuid="e66e0078-7015-5450-92f7-15fbd957f2ae")),
    Dependency(PackageSpec(name="OpenBLAS32_jll", uuid="656ef2d0-ae68-5445-9ca0-591084a874a2")),
    Dependency(PackageSpec(name="OpenBLAS_jll", uuid="4536629a-c528-5b80-bd46-f80d51c5b363")),
    Dependency(PackageSpec(name="Hwloc_jll", uuid="e33a78d0-f292-5ffc-b300-72abe9b543c8")),
    Dependency(PackageSpec(name="METIS_jll", uuid="d00139f3-1899-568f-a2f0-47f597d42d70")),
    Dependency(PackageSpec(name="HSL_jll", uuid="017b0a0e-03f4-516a-9b91-836bbd1904dd")),
    Dependency(PackageSpec(name="CUTEst_jll", uuid="bb5f6f25-f23d-57fd-8f90-3ef7bad1d825"), compat="2.6.0"),
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; preferred_gcc_version=v"9.1.0", julia_compat="1.6")
