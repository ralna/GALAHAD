libgalahad_src += files('uls.F90')
if build_ciface
  libgalahad_c_src += files('C/uls_ciface.F90')
endif

galahad_c_tests += [['uls', 'ulst_c', files('C/ulst.c')],
                    ['uls', 'ulstf_c', files('C/ulstf.c')]]

galahad_examples += [['ulss', files('ulss.f90')]]
