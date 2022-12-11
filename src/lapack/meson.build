libgalahad_blas_src = files('blas.f')
libgalahad_src += files('blas_interface.F90')

libgalahad_lapack_src = files('lapack.f')
if fc.get_id() == 'nagfor'
  libgalahad_lapack_src += files('noieeeck.f')
else
  libgalahad_lapack_src += files('ieeeck.f')
endif
libgalahad_src += files('lapack_interface.F90')
