libgalahad_src += files('ugo.F90')
if build_ciface
  libgalahad_c_src += files('C/ugo_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useugo.F90', 'runugo_sif.F90')
endif
galahad_c_examples += [['ugos_c', files('C/ugos.c')]]
