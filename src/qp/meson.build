libgalahad_src += files('qp.F90')

galahad_binaries += [['inqp', files('inqp.F90')]]

if libcutest.found()
  libgalahad_cutest_src += files('useqp.F90', 'runqp_sif.F90')
endif

galahad_tests += [['qp', 'qpt', files('qpt.F90')]]

galahad_examples += [['qps', files('qps.f90')]]
