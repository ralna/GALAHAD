libgalahad_src += files('uls.F90')

libgalahad_c_src += files('C/uls_ciface.F90')

libgalahad_python_src += [['uls', files('Python/uls_pyiface.c',
                                        '../gls/Python/gls_pyiface.c')]]

galahad_tests += [['uls', 'ulst', files('ulst.F90')],
                  ['uls', 'ulsti', files('ulsti.F90')]]

galahad_c_tests += [['uls', 'ulst_c', files('C/ulst.c')],
                    ['uls', 'ulstf_c', files('C/ulstf.c')]]

galahad_python_tests += [['uls', 'uls_py', files('Python/test_uls.py')]]

galahad_examples += [['ulss', files('ulss.f90')]]
