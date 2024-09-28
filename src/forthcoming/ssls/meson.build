libgalahad_src += files('ssls.F90')

#libgalahad_c_src += files('C/ssls_ciface.F90')

#libgalahad_python_src += [['ssls', files('Python/ssls_pyiface.c',
                                          '../sls/Python/sls_pyiface.c',
                                          '../sils/Python/sils_pyiface.c',
                                          '../uls/Python/uls_pyiface.c',
                                          '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usessls.F90')

galahad_cutest_binaries += [['runssls_sif', files('runssls_sif.F90')]]

galahad_tests += [['ssls', 'sslst', files('sslst.F90')],
                  ['ssls', 'sslsti', files('sslsti.F90')]]

#galahad_c_tests += [['ssls', 'sslst_c', files('C/sslst.c')],
                    ['ssls', 'sslstf_c', files('C/sslstf.c')]]

#galahad_python_tests += [['ssls', 'ssls_py', files('Python/test_ssls.py')]]

galahad_examples += [['sslss', files('sslss.f90')]]
