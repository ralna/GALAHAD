libgalahad_src += files('blas_inter.F90', 'lapack_inter.F90')

if not libblas.found()
  warning('building our own BLAS v3.9.1; consider providing an optimized BLAS library')
  libgalahad_src += files('blas.f90')
endif

if not liblapack.found()
  warning('building our own LAPACK v3.9.1; consider providing an optimized LAPACK library')
  libgalahad_src += files('lapack.f90')
  if fc.get_id() == 'nagfor'
    libgalahad_src += files('noieeeck.f90')
  else
    libgalahad_src += files('ieeeck.f90')
  endif
endif
