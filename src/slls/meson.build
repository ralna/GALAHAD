libgalahad_src += files('slls.F90')

libgalahad_c_src += files('C/slls_ciface.F90')

libgalahad_python_src += [['slls', files('Python/slls_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c','../sls/Python/sls_pyiface.c',
                                         '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c',
                                         '../gls/Python/gls_pyiface.c','../convert/Python/convert_pyiface.c')]]

libgalahad_cutest_src += files('useslls.F90')

galahad_cutest_binaries += [['runslls_sif', files('runslls_sif.F90')]]

galahad_tests += [['slls', 'sllst', files('sllst.F90')],
                  ['slls', 'sllsti', files('sllsti.F90')]]

galahad_c_tests += [['slls', 'sllst_c', files('C/sllst.c')],
                    ['slls', 'sllstf_c', files('C/sllstf.c')]]

galahad_python_tests += [['slls', 'slls_py', files('Python/test_slls.py')]]

galahad_examples += [['sllss', files('sllss.f90')],
                     ['sllss2', files('sllss2.f90')],
                     ['sllss3', files('sllss3.f90')],
                     ['sllss4', files('sllss4.f90')],
                     ['sllss5', files('sllss5.f90')],
                     ['sllss6', files('sllss6.f90')],
                     ['sllss7', files('sllss7.f90')]]
