libgalahad_src += files('slls.F90')
if build_ciface
  libgalahad_c_src += files('C/slls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useslls.F90', 'runslls_sif.F90')
endif

galahad_c_tests += [['slls', 'sllst_c', files('C/sllst.c')],
                    ['slls', 'sllstf_c', files('C/sllstf.c')]]

galahad_examples += [['sllss', files('sllss.f90')],
                     ['sllss2', files('sllss2.f90')],
                     ['sllss3', files('sllss3.f90')],
                     ['sllss4', files('sllss4.f90')],
                     ['sllss5', files('sllss5.f90')],
                     ['sllss6', files('sllss6.f90')],
                     ['sllss7', files('sllss7.f90')]]
