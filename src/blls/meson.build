libgalahad_src += files('blls.F90')
if build_ciface
  libgalahad_c_src += files('C/blls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useblls.F90', 'runblls_sif.F90')
endif

galahad_tests += [['blls', 'bllst', files('bllst.F90')],
                  ['blls', 'bllsti', files('bllsti.F90')]]

galahad_c_tests += [['blls', 'bllst_c', files('C/bllst.c')],
                    ['blls', 'bllstf_c', files('C/bllstf.c')]]

galahad_examples += [['bllss', files('bllss.f90')],
                     ['bllss2', files('bllss2.f90')],
                     ['bllss3', files('bllss3.f90')]]
