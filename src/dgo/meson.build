libgalahad_src += files('dgo.F90')
if build_ciface
  libgalahad_c_src += files('C/dgo_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usedgo.F90', 'rundgo_sif.F90')
endif
