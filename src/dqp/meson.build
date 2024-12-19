libgalahad_src += files('dqp.F90')

galahad_binaries += [['indqp', files('indqp.F90')]]

libgalahad_c_src += files('C/dqp_ciface.F90')

libgalahad_python_src += [['dqp', files('Python/dqp_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../sbls/Python/sbls_pyiface.c','../gltr/Python/gltr_pyiface.c',
                                        '../scu/Python/scu_pyiface.c','../rpd/Python/rpd_pyiface.c','../sils/Python/sils_pyiface.c',
                                        '../lms/Python/lms_pyiface.c','../fdc/Python/fdc_pyiface.c','../uls/Python/uls_pyiface.c',
                                        '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usedqp.F90')

galahad_cutest_binaries += [['rundqp_sif', files('rundqp_sif.F90')]]

galahad_tests += [['dqp', 'dqpt', files('dqpt.F90')],
                  ['dqp', 'dqpti', files('dqpti.F90')]]

galahad_c_tests += [['dqp', 'dqpt_c', files('C/dqpt.c')],
                    # ['dqp', 'dqpt2_c', files('C/dqpt2.c')],
                    ['dqp', 'dqptf_c', files('C/dqptf.c')]]

galahad_python_tests += [['dqp', 'dqp_py', files('Python/test_dqp.py')]]

galahad_examples += [['dqps', files('dqps.f90')],
                     ['dqps2', files('dqps2.f90')],
                     ['dqps3', files('dqps3.f90')]]
