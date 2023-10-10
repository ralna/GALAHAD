libgalahad_src += files('slls.F90')
if build_ciface
  libgalahad_c_src += files('C/slls_ciface.F90')
endif

libgalahad_cutest_src += files('useslls.F90')

galahad_cutest_binaries += [['runslls_sif', files('runslls_sif.F90')]]

galahad_tests += [['slls', 'sllst', files('sllst.F90')],
                  ['slls', 'sllsti', files('sllsti.F90')]]

galahad_c_tests += [['slls', 'sllst_c', files('C/sllst.c')],
                    ['slls', 'sllstf_c', files('C/sllstf.c')]]

galahad_examples += [['sllss', files('sllss.f90')],
                     ['sllss2', files('sllss2.f90')],
                     ['sllss3', files('sllss3.f90')],
                     ['sllss4', files('sllss4.f90')],
                     ['sllss5', files('sllss5.f90')],
                     ['sllss6', files('sllss6.f90')],
                     ['sllss7', files('sllss7.f90')]]
