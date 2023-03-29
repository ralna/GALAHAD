libgalahad_src += files('rqs.F90')
if build_ciface
  libgalahad_c_src += files('C/rqs_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('userqs.F90', 'runrqs_sif.F90')
endif

galahad_tests += [['rqs', 'rqst', files('rqst.F90')],
                  ['rqs', 'rqsti', files('rqsti.F90')]]

galahad_c_tests += [['rqs', 'rqst_c', files('C/rqst.c')],
                    ['rqs', 'rqstf_c', files('C/rqstf.c')]]

galahad_examples += [['rqss', files('rqss.f90')],
                     ['rqss2', files('rqss2.f90')]]
