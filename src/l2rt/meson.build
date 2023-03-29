libgalahad_src += files('l2rt.F90')
if build_ciface
  libgalahad_c_src += files('C/l2rt_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usel2rt.F90', 'runl2rt_sif.F90')
endif

galahad_c_tests += [['l2rt', 'l2rtt_c', files('C/l2rtt.c')]]

galahad_examples += [['l2rts', files('l2rts.f90')],
                     ['l2rts2', files('l2rts2.f90')]]
