libgalahad_src += files('blls.F90')

libgalahad_c_src += files('C/blls_ciface.F90')


libgalahad_python_src += [['blls', files('Python/blls_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c','../convert/Python/convert_pyiface.c','../sls/Python/sls_pyiface.c',
                                         '../uls/Python/uls_pyiface.c','../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c',
                                         '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('useblls.F90')

galahad_cutest_binaries += [['runblls_sif', files('runblls_sif.F90')]]

galahad_tests += [['blls', 'bllst', files('bllst.F90')],
                  ['blls', 'bllsti', files('bllsti.F90')]]

galahad_c_tests += [['blls', 'bllst_c', files('C/bllst.c')],
                    ['blls', 'bllstf_c', files('C/bllstf.c')]]

galahad_python_tests += [['blls', 'blls_py', files('Python/test_blls.py')]]

galahad_examples += [['bllss', files('bllss.f90')],
                     ['bllss2', files('bllss2.f90')],
                     ['bllss3', files('bllss3.f90')]]
