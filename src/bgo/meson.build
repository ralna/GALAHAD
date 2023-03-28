libgalahad_src += files('bgo.F90')
if build_ciface
  libgalahad_c_src += files('C/bgo_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usebgo.F90', 'runbgo_sif.F90')
endif
