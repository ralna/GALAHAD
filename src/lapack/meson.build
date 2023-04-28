libgalahad_src += files('blas_interface.F90', 'lapack_interface.F90')

if not libblas.found()
  warning('building our own BLAS; consider providing an optimized BLAS library')
  libgalahad_src += files('blas.f')
endif

if not liblapack.found()
  warning('building our own LAPACK; consider providing an optimized LAPACK library')
  libgalahad_src += files('lapack.f')
  if fc.get_id() == 'nagfor'
    libgalahad_src += files('noieeeck.f')
  else
    libgalahad_src += files('ieeeck.f')
  endif
endif
