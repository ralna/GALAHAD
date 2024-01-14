libgalahad_src += files('lsqp.F90')

libgalahad_c_src += files('C/lsqp_ciface.F90')

libgalahad_python_src += [['lsqp', files('Python/lsqp_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c','../gltr/Python/gltr_pyiface.c','../sls/Python/sls_pyiface.c',
                                         '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c',
                                         '../fdc/Python/fdc_pyiface.c')]]

galahad_tests += [['lsqp', 'lsqpt', files('lsqpt.F90')],
                  ['lsqp', 'lsqpti', files('lsqpti.F90')]]

galahad_c_tests += [['lsqp', 'lsqpt_c', files('C/lsqpt.c')],
                    ['lsqp', 'lsqptf_c', files('C/lsqptf.c')]]

galahad_python_tests += [['lsqp', 'lsqp_py', files('Python/test_lsqp.py')]]

galahad_examples += [['lsqps', files('lsqps.f90')],
                     ['lsqps2', files('lsqps2.f90')]]
