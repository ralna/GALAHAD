blas_src = ['src/lapack/blas.f']
libgalahad_blas_src += ['src/lapack/blas_interface.f90']

lapack_src = ['src/lapack/lapack.f']
if fc.get_id() == 'nagfor'
  lapack_src += ['src/lapack/noieeeck.f']
else
  lapack_src += ['src/lapack/ieeeck.f']
endif
libgalahad_lapack_src += ['src/lapack/lapack_interface.f90']
