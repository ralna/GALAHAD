libgalahad_src += files('bqp.F90')

galahad_binaries += [['inbqp', files('inbqp.F90')]]

libgalahad_c_src += files('C/bqp_ciface.F90')

libgalahad_python_src += [['bqp', files('Python/bqp_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c','../gltr/Python/gltr_pyiface.c','../sls/Python/sls_pyiface.c',
                                        '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usebqp.F90')

galahad_cutest_binaries += [['runbqp_sif', files('runbqp_sif.F90')]]

galahad_tests += [['bqp', 'bqpt', files('bqpt.F90')],
                  ['bqp', 'bqpti', files('bqpti.F90')]]

galahad_c_tests += [['bqp', 'bqpt_c', files('C/bqpt.c')],
                    ['bqp', 'bqptf_c', files('C/bqptf.c')]]

galahad_python_tests += [['bqp', 'bqp_py', files('Python/test_bqp.py')]]

galahad_examples += [['bqps', files('bqps.f90')],
                     ['bqps2', files('bqps2.f90')],
                     ['bqps3', files('bqps3.f90')]]
