blas_src = ['src/lapack/blas.f', 'src/lapack/blas_interface.f90']
libgalahad_blas_src += blas_src

lapack_src = ['src/lapack/lapack.f', 'src/lapack/lapack_interface.f90']
if meson.get_compiler('fortran').get_id() == 'nagfor'
  lapack_src += ['src/lapack/noieeeck.f']
else
  lapack_src += ['src/lapack/ieeeck.f']
endif
libgalahad_lapack_src += lapack_src
