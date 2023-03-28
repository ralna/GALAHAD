libgalahad_src += files('lqt.F90')
if libcutest.found()
  libgalahad_cutest_src += files('uselqt.F90', 'runlqt_sif.F90')
endif
galahad_examples += [['lqts', files('lqts.f90')],
                     ['lqts2', files('lqts2.f90')]]
