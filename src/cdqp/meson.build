libgalahad_src += files('cdqp.F90')
bingalahad_src += [['incdqp', files('incdqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/cdqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usecdqp.F90', 'runcdqp_sif.F90')
endif
galahad_examples += [['cdqps', files('cdqps.f90')]]
