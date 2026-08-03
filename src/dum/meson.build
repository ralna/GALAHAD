libgalahad_src += files('mpi.F90')
libgalahad_cutest_dummy_src = files('cutest_dummy.F90')

if not libwsmp.found()
  libgalahad_single_src += files('wsmp.F90')
  libgalahad_double_src += files('wsmp.F90')
endif
libgalahad_quadruple_src += files('wsmp.F90')

if not libpardiso.found()
  libgalahad_single_src += files('pardiso.F90')
  libgalahad_double_src += files('pardiso.F90')
endif
libgalahad_quadruple_src += files('pardiso.F90')

libgalahad_single_src += files('spmf_enums.F90', 'pastixf_enums.F90')
libgalahad_double_src += files('spmf_enums.F90', 'pastixf_enums.F90')
if not (libpastixf.found() and libspmf.found())
  libgalahad_single_src += files('spmf.F90', 'pastixf.F90')
  libgalahad_double_src += files('spmf.F90', 'pastixf.F90')
endif
libgalahad_quadruple_src += files('spmf.F90', 'spmf_enums.F90', 'pastixf.F90', 'pastixf_enums.F90')

if not libsmumps.found()
  libgalahad_single_src += files('mumps.F90')
endif
if not libdmumps.found()
  libgalahad_double_src += files('mumps.F90')
endif
libgalahad_quadruple_src += files('mumps.F90')

if not (libblas_name == 'mkl_rt' or liblapack_name == 'mkl_rt')
  libgalahad_single_src += files('mkl_pardiso.F90')
  libgalahad_double_src += files('mkl_pardiso.F90')
endif
libgalahad_quadruple_src += files('mkl_pardiso.F90')
