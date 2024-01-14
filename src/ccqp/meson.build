libgalahad_src += files('ccqp.F90')

galahad_binaries += [['inccqp', files('inccqp.F90')]]

libgalahad_c_src += files('C/ccqp_ciface.F90')

libgalahad_cutest_src += files('useccqp.F90')

galahad_cutest_binaries += [['runccqp_sif', files('runccqp_sif.F90')]]

galahad_tests += [['ccqp', 'ccqpt', files('ccqpt.F90')],
                  ['ccqp', 'ccqpti', files('ccqpti.F90')]]

galahad_c_tests += [['ccqp', 'ccqpt_c', files('C/ccqpt.c')],
                    ['ccqp', 'ccqptf_c', files('C/ccqptf.c')]]

galahad_examples += [['ccqps', files('ccqps.f90')],
                     ['ccqps2', files('ccqps2.f90')],
                     ['ccqps3', files('ccqps3.f90')]]
