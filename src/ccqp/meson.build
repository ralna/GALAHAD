libgalahad_src += files('ccqp.F90')
galahad_binaries += [['inccqp', files('inccqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/ccqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useccqp.F90', 'runccqp_sif.F90')
endif

galahad_c_tests += [['ccqp', 'ccqpt_c', files('C/ccqpt.c')],
                    ['ccqp', 'ccqptf_c', files('C/ccqptf.c')]]

galahad_examples += [['ccqps', files('ccqps.f90')],
                     ['ccqps2', files('ccqps2.f90')],
                     ['ccqps3', files('ccqps3.f90')]]
