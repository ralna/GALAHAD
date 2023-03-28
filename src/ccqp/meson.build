libgalahad_src += files('ccqp.F90')
bingalahad_src += [['inccqp', files('inccqp.F90')]]
libgalahad_c_src += files('C/ccqp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('useccqp.F90', 'runccqp_sif.F90')
endif
