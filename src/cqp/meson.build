libgalahad_src += files('cqp.F90')
bingalahad_src += [['incqp', files('incqp.F90')]]
libgalahad_c_src += files('C/cqp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usecqp.F90', 'runcqp_sif.F90')
endif
