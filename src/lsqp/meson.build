libgalahad_src += files('lsqp.F90')
if build_ciface
  libgalahad_c_src += files('C/lsqp_ciface.F90')
endif

galahad_tests += [['lsqp', 'lsqpt', files('lsqpt.F90')],
                  ['lsqp', 'lsqpti', files('lsqpti.F90')]]

galahad_c_tests += [['lsqp', 'lsqpt_c', files('C/lsqpt.c')],
                    ['lsqp', 'lsqptf_c', files('C/lsqptf.c')]]

galahad_examples += [['lsqps', files('lsqps.f90')],
                     ['lsqps2', files('lsqps2.f90')]]
