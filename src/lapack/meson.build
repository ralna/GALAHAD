libgalahad_blas_src = ['src/lapack/blas.f']
libgalahad_src += ['src/lapack/blas_interface.f90']

libgalahad_lapack_src = ['src/lapack/lapack.f']
if fc.get_id() == 'nagfor'
  libgalahad_lapack_src += ['src/lapack/noieeeck.f']
else
  libgalahad_lapack_src += ['src/lapack/ieeeck.f']
endif
libgalahad_src += ['src/lapack/lapack_interface.f90']
