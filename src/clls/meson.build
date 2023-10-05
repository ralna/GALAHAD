libgalahad_src += files('clls.F90')

if build_ciface
  libgalahad_c_src += files('C/clls_ciface.F90')
endif

libgalahad_cutest_src += files('useclls.F90', 'runclls_sif.F90')

galahad_tests += [['clls', 'cllst', files('cllst.F90')],
                  ['clls', 'cllsti', files('cllsti.F90')]]

galahad_c_tests += [['clls', 'cllst_c', files('C/cllst.c')],
                    ['clls', 'cllstf_c', files('C/cllstf.c')]]

galahad_examples += [['cllss', files('cllss.f90')],
                     ['cllss2', files('cllss2.f90')],
                     ['cllss3', files('cllss3.f90')]]
