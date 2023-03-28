libgalahad_src += files('dgo.F90')
if build_ciface
  libgalahad_c_src += files('C/dgo_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usedgo.F90', 'rundgo_sif.F90')
endif
galahad_examples += [['dgos', files('dgos.f90')],
                     ['dgos2', files('dgos2.f90')]]
galahad_c_examples += [['dgos_c', files('C/dgos.c')],
                       ['dgos2_c', files('C/dgos2.c')],
                       ['dgos3_c', files('C/dgos3.c')],
                       ['dgos4_c', files('C/dgos4.c')]]
