libgalahad_src += files('glrt.F90')
if build_ciface
  libgalahad_c_src += files('C/glrt_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useglrt.F90', 'runglrt_sif.F90')
endif

galahad_tests += [['glrt', 'glrtt', files('glrtt.F90')]]

galahad_c_tests += [['glrt', 'glrtt_c', files('C/glrtt.c')]]

galahad_examples += [['glrts', files('glrts.f90')]]
