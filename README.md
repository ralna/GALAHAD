# GALAHAD [![][license-shield]][license] [![][joss-shield]][joss] [![codecov][codecov-img]][codecov-url]
GALAHAD is a library of modern Fortran packages for nonlinear optimization with C, Python, Julia and MATLAB interfaces. It contains packages for general constrained and unconstrained optimization, linear and quadratic programming, nonlinear least-squares fitting and global optimization, as well as those for solving a large variety of basic optimization subproblems.

## Documentation
More information on the packages in GALAHAD can be found at https://www.galahad.rl.ac.uk.

All major GALAHAD packages are documented in Fortran, C, Python and Julia:

* [Fortran Documentation (PDF)](https://ralna.github.io/galahad_docs/pdf/Fortran)
* [C Documentation (HTML)](https://ralna.github.io/galahad_docs/html/C)
* [Python Documentation (HTML)](https://ralna.github.io/galahad_docs/html/Python)
* [Julia Documentation (HTML)](https://ralna.github.io/galahad_docs/html/Julia)

Help files are provided for MATLAB functions.

## Installation

### Precompiled Fortran/C libraries and executables
We provide precompiled GALAHAD libraries and executables in the [releases tab](https://github.com/ralna/galahad/releases/latest/) for Linux (x64 and aarch64), macOS (x64 and aarch64), and Windows (x64).

On some platforms, the dynamic linker needs to know where to look for libraries at runtime.
You might need to set the following environment variables:

- `LD_LIBRARY_PATH` on Linux
- `DYLD_LIBRARY_PATH` or `DYLD_FALLBACK_LIBRARY_PATH` on macOS
- `PATH` on Windows

These variables should include the directory where you extracted the libraries.
For all platforms, the environment variable `PATH` is needed to locate the executables.

### Precompiled Julia Interface
We provide a precompiled Julia interface for most platforms, please see [GALAHAD.jl](https://github.com/ralna/GALAHAD/tree/master/GALAHAD.jl) and the associated documentation.

### Precompiled Python Interface
We provide a precompiled Python interface for Linux, macOS (Intel & Silicon), and Windows that [can be installed from PyPI](https://pypi.org/project/galahad-optrove/):
```
pip install galahad-optrove
```

### Installation from source 
GALAHAD can be installed from source using the [Meson build system](https://mesonbuild.com) (all commands below are to be run from the top of the source tree):

```
meson setup builddir -Dtests=true
meson compile -C builddir
meson install -C builddir
meson test -C builddir
```

For more comprehensive Meson options (`-Doption=value`), including how to specify paths to various libraries and packages, please see [meson_options.txt](https://github.com/ralna/GALAHAD/blob/master/meson_options.txt) and [README.meson](https://github.com/ralna/GALAHAD/blob/master/README.meson). We give some examples below for the most important Meson options.

GALAHAD supports a large number of optional software packages for enhanced functionality, the most important of these are:

#### BLAS/LAPACK
By default GALAHAD will build with [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) if it can locate it (otherwise you may need to pass the OpenBLAS paths via the `libblas_path` and `liblapack_path` options to `meson setup`). You may also wish to use a vendor-specific BLAS/LAPACK implementation such as one of the following:

* [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
* [AMD AOCL](https://www.amd.com/en/developer/aocl/dense.html)

Please see [README.meson](https://github.com/ralna/GALAHAD/blob/master/README.meson) for instructions on how to tell Meson where to find these optional dependencies.

#### Linear Solvers
By default GALAHAD will build the [SSIDS linear solver](https://github.com/ralna/spral), other alternative linear solvers are:

* [HSL](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL)
* [UMFPACK](https://people.engr.tamu.edu/davis/suitesparse.html)
* [PARDISO](https://panua.ch/pardiso/)
* [PaStiX](https://solverstack.gitlabpages.inria.fr/pastix/)
* [MUMPS](https://mumps-solver.org/index.php)

Please see [README.meson](https://github.com/ralna/GALAHAD/blob/master/README.meson) for instructions on how to tell Meson where to find these optional dependencies.

If you use the precompiled GALAHAD libraries, the MUMPS libraries are also provided.
For libHSL, you can replace the dummy shared libraries `libhsl_subset` and `libhsl_subset_64`
shipped with GALAHAD with the official ones provided by the precompiled libHSL libraries.
This requires at least GALAHAD v5.4.0 and libHSL v2025.7.21.

#### CUTEst Test Collection
GALAHAD can use optimization test problems from the [CUTEst test collection](https://github.com/ralna/CUTEst/blob/master/doc/README). For example, to link GALAHAD with double precision CUTEst:

```
meson setup builddir -Dlibcutest_path=/path/to/CUTEst/lib -Dlibcutest_modules=/path/to/CUTEst/modules -Dsingle=false
meson compile -C builddir
meson install -C builddir
```

GALAHAD can similarly be linked with the single or quadruple precision variants of CUTEst.
For more details, refer to the file [meson_options.txt](https://github.com/ralna/GALAHAD/blob/master/meson_options.txt).

**Note:** only the shared libraries of CUTEst are supported when compiling GALAHAD with Meson.
Please follow the [instructions](https://github.com/ralna/CUTEst?tab=readme-ov-file#new-approach-with-shared-libraries-and-trampolines) to set up CUTEst accordingly.

#### C Interface
To install the C interface using the [Meson build system](https://mesonbuild.com):
```
meson setup builddir -Dciface=true
meson compile -C builddir
meson install -C builddir
meson test -C builddir --suite=C
```

#### Python Interface
To install the Python interface using the [Meson build system](https://mesonbuild.com):
```
meson setup builddir -Dpythoniface=true -Dpython.install_env=auto
meson compile -C builddir
meson install -C builddir
meson test -C builddir --suite=Python
```

#### Julia Interface
Please see [GALAHAD.jl](https://github.com/ralna/GALAHAD/tree/master/GALAHAD.jl) and the associated documentation.

#### MATLAB Interface
Please see [README.matlab](https://github.com/ralna/GALAHAD/blob/master/doc/README.matlab) and the instructions provided there.

## Integrated installation via make

GALAHAD can also be installed via the `make` command as part of the Optrove 
optimization eco-system that also includes 
[CUTEst](https://github.com/ralna/CUTEst), 
[SIFDecode](https://github.com/ralna/SIFDecode) and
[ARCHDefs](https://github.com/ralna/ARCHDefs). 
This has the advantage of providing scripts to run CUTEst examples
directly from GALAHAD and allowing calls from Matlab, but suffers from 
considerably longer build times.

To use this variant, follow the instructions in the GALAHAD
[wiki](https://github.com/ralna/GALAHAD/wiki).

[license-shield]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg 
[license]: https://opensource.org/licenses/BSD-3-Clause
[joss-shield]: https://joss.theoj.org/papers/10.21105/joss.04882/status.svg
[joss]: https://doi.org/10.21105/joss.04882
[codecov-img]: https://codecov.io/gh/ralna/GALAHAD/branch/master/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/ralna/GALAHAD
