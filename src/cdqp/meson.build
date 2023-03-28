libgalahad_src += files('cdqp.F90')
bingalahad_src += [['incdqp', files('incdqp.F90')]]
libgalahad_c_src += files('C/cdqp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usecdqp.F90', 'runcdqp_sif.F90')
endif
