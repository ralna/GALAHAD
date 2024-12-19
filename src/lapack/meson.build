libgalahad_src += files('blas_inter.F90', 'lapack_inter.F90')
libblas_src += files('blas.f90')
liblapack_src += files('lapack.f90')
# if fc.get_id() == 'nagfor'
#   liblapack_src += files('noieeeck.f90')
# else
#   liblapack_src += files('ieeeck.f90')
# endif

if not libblas.found()
  warning('building our own BLAS v3.9.1; consider providing an optimized BLAS library')
  libgalahad_single_src += libblas_src
  libgalahad_double_src += libblas_src
endif
libgalahad_quadruple_src += libblas_src

if not liblapack.found()
  warning('building our own LAPACK v3.9.1; consider providing an optimized LAPACK library')
  libgalahad_single_src += liblapack_src
  libgalahad_double_src += liblapack_src
endif
libgalahad_quadruple_src += liblapack_src
