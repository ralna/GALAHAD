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

if not libspmf.found()
  libgalahad_single_src += files('spmf.F90', 'spmf_enums.F90')
  libgalahad_double_src += files('spmf.F90', 'spmf_enums.F90')
endif
libgalahad_quadruple_src += files('spmf.F90', 'spmf_enums.F90')

if not libpastixf.found()
  libgalahad_single_src += files('pastixf.F90', 'pastixf_enums.F90')
  libgalahad_double_src += files('pastixf.F90', 'pastixf_enums.F90')
endif
libgalahad_quadruple_src += files('pastixf.F90', 'pastixf_enums.F90')

if not libsmumps.found()
  libgalahad_single_src += files('mumps.F90')
  galahad_headers += files('include/smumps_struc.h')
endif

if not libdmumps.found()
  libgalahad_double_src += files('mumps.F90')
  galahad_headers += files('include/dmumps_struc.h')
endif

libgalahad_quadruple_src += files('mumps.F90')
galahad_headers += files('include/qmumps_struc.h')

if not (libblas_name == 'mkl_rt' or liblapack_name == 'mkl_rt')
  libgalahad_single_src += files('mkl_pardiso.F90')
  libgalahad_double_src += files('mkl_pardiso.F90')
endif
libgalahad_quadruple_src += files('mkl_pardiso.F90')

if not build_ssids
  libgalahad_single_src += files('ssidss.F90')
  libgalahad_double_src += files('ssidsd.F90')
  libgalahad_quadruple_src += files('ssidsq.F90')
  libgalahad_c_src += files('C/ssids_ciface.F90')
endif
