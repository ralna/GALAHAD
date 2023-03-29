libgalahad_src += files('lancelot_simple.F90')
if libcutest.found()
  libgalahad_cutest_src += files('run_lancelot_simple.F90')
endif

galahad_tests += [['lancelot_simple', 'lancelot_simplet', files('lancelot_simplet.F90')]]

galahad_examples += [['lancelot_simples', files('lancelot_simples.f90')]]
