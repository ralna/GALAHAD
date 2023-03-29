libgalahad_src += files('rpd.F90')
if build_ciface
  libgalahad_c_src += files('C/rpd_ciface.F90')
endif

galahad_tests += [['rpd', 'rpdt', files('rpdt.F90')],
                  ['rpd', 'rpdti', files('rpdti.F90')]]

galahad_c_tests += [['rpd', 'rpdt_c', files('C/rpdt.c')],
                    ['rpd', 'rpdtf_c', files('C/rpdtf.c')]]

galahad_examples += [['rpds', files('rpds.f90')]]
