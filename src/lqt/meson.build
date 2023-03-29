libgalahad_src += files('lqt.F90')
if libcutest.found()
  libgalahad_cutest_src += files('uselqt.F90', 'runlqt_sif.F90')
endif

galahad_tests += [['lqt', 'lqtt', files('lqtt.F90')],
                  ['lqt', 'lqtt2', files('lqtt2.F90')]]

galahad_examples += [['lqts', files('lqts.f90')],
                     ['lqts2', files('lqts2.f90')]]
