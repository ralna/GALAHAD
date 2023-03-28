libgalahad_src += files('dqp.F90')
bingalahad_src += [['indqp', files('indqp.F90')]]
libgalahad_c_src += files('C/dqp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usedqp.F90', 'rundqp_sif.F90')
endif
