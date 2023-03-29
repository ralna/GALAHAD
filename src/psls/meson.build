libgalahad_src += files('psls.F90')
if build_ciface
  libgalahad_c_src += files('C/psls_ciface.F90')
endif

galahad_tests += [['psls', 'pslst', files('pslst.F90')],
                  ['psls', 'pslsti', files('pslsti.F90')]]

galahad_c_tests += [['psls', 'pslst_c', files('C/pslst.c')],
                    ['psls', 'pslstf_c', files('C/pslstf.c')]]

galahad_examples += [['pslss', files('pslss.f90')]]
