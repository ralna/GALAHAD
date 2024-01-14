libgalahad_src += files('eqp.F90')

libgalahad_c_src += files('C/eqp_ciface.F90')

libgalahad_python_src += [['eqp', files('Python/eqp_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../sbls/Python/sbls_pyiface.c','../gltr/Python/gltr_pyiface.c',
                                        '../fdc/Python/fdc_pyiface.c','../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c',
                                        '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('useeqp.F90')

galahad_cutest_binaries += [['runeqp_sif', files('runeqp_sif.F90')]]

galahad_tests += [['eqp', 'eqpt', files('eqpt.F90')],
                  ['eqp', 'eqpti', files('eqpti.F90')]]

galahad_c_tests += [['eqp', 'eqpt_c', files('C/eqpt.c')],
                    ['eqp', 'eqptf_c', files('C/eqptf.c')]]

galahad_python_tests += [['eqp', 'eqp_py', files('Python/test_eqp.py')]]

galahad_examples += [['eqps', files('eqps.f90')]]
