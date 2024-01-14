libgalahad_src += files('lpb.F90')

galahad_binaries += [['inlpb', files('inlpb.F90')]]

libgalahad_c_src += files('C/lpb_ciface.F90')

libgalahad_python_src += [['lpb', files('Python/lpb_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c','../sls/Python/sls_pyiface.c','../uls/Python/uls_pyiface.c',
                                        '../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c','../gls/Python/gls_pyiface.c',
                                        '../fdc/Python/fdc_pyiface.c','../scu/Python/scu_pyiface.c','../fit/Python/fit_pyiface.c',
                                        '../ir/Python/ir_pyiface.c','../cro/Python/cro_pyiface.c','../rpd/Python/rpd_pyiface.c',
                                        '../lms/Python/lms_pyiface.c',)]]

libgalahad_cutest_src += files('uselpb.F90')

galahad_cutest_binaries += [['runlpb_sif', files('runlpb_sif.F90')]]

galahad_tests += [['lpb', 'lpbt', files('lpbt.F90')],
                  ['lpb', 'lpbti', files('lpbti.F90')]]

galahad_c_tests += [['lpb', 'lpbt_c', files('C/lpbt.c')],
                    ['lpb', 'lpbtf_c', files('C/lpbtf.c')]]

galahad_python_tests += [['lpb', 'lpb_py', files('Python/test_lpb.py')]]

galahad_examples += [['lpbs', files('lpbs.f90')]]
