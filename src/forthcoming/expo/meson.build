libgalahad_src += files('expo.F90')

libgalahad_c_src += files('C/expo_ciface.F90')

# libgalahad_python_src += [['expo', files('Python/expo_pyiface.c',
#                                          '../../ssls/Python/ssls_pyiface.c',
#                                          '../../bsc/Python/bsc_pyiface.c',
#                                          '../../tru/Python/tru_pyiface.c')]]

libgalahad_cutest_src += files('useexpo.F90')

galahad_cutest_binaries += [['runexpo_sif', files('runexpo_sif.F90')]]

galahad_tests += [#['expo', 'expot', files('expot.F90')],
                  ['expo', 'expoti', files('expoti.F90')]]

# galahad_c_tests += [['expo', 'expot_c', files('C/expot.c')],
#                     ['expo', 'expotf_c', files('C/expotf.c')]]

# galahad_python_tests += [['expo', 'expo_py', files('Python/test_expo.py')]]

galahad_examples += [['expos', files('expos.f90')]]

#galahad_examples += [['expos', files('expos.f90')],
#                     ['expos2', files('expos2.f90')],
#                     ['expos3', files('expos3.f90')]]
