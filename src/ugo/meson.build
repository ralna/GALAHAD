libgalahad_src += files('ugo.F90')
if build_ciface
  libgalahad_c_src += files('C/ugo_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useugo.F90', 'runugo_sif.F90')
endif

galahad_c_tests += [['ugo', 'ugot_c', files('C/ugot.c')]]

galahad_examples += [['ugos', files('ugos.f90')],
                     ['ugos2', files('ugos2.f90')],
                     ['ugos3', files('ugos3.f90')]]

galahad_c_examples += [['ugos_c', files('C/ugos.c')]]
