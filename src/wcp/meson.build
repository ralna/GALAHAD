libgalahad_src += files('wcp.F90')
if build_ciface
  libgalahad_c_src += files('C/wcp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usewcp.F90', 'runwcp_sif.F90')
endif

galahad_tests += [['wcp', 'wcpt', files('wcpt.F90')],
                  ['wcp', 'wcpti', files('wcpti.F90')]]

galahad_c_tests += [['wcp', 'wcpt_c', files('C/wcpt.c')],
                    ['wcp', 'wcptf_c', files('C/wcptf.c')]]

galahad_examples += [['wcps', files('wcps.f90')]]
