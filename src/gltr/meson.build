libgalahad_src += files('gltr.F90')
if build_ciface
  libgalahad_c_src += files('C/gltr_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usegltr.F90', 'rungltr_sif.F90')
endif
