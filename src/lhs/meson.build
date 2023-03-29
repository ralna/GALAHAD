libgalahad_src += files('lhs.F90')
if build_ciface
  libgalahad_c_src += files('C/lhs_ciface.F90')
endif

galahad_tests += [['lhs', 'lhst', files('lhst.F90')]]

galahad_c_tests += [['lhs', 'lhst_c', files('C/lhst.c')]]

galahad_examples += [['lhss', files('lhss.f90')]]

galahad_c_examples += [['lhss_c', files('C/lhss.c')]]
