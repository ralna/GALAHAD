libgalahad_src += files('psls.F90')

libgalahad_c_src += files('C/psls_ciface.F90')

libgalahad_python_src += [['psls', files('Python/psls_pyiface.c',
                                         '../sls/Python/sls_pyiface.c','../sils/Python/sils_pyiface.c')]]

galahad_tests += [['psls', 'pslst', files('pslst.F90')],
                  ['psls', 'pslsti', files('pslsti.F90')]]

galahad_c_tests += [['psls', 'pslst_c', files('C/pslst.c')],
                    ['psls', 'pslstf_c', files('C/pslstf.c')]]

galahad_python_tests += [['psls', 'psls_py', files('Python/test_psls.py')]]

galahad_examples += [['pslss', files('pslss.f90')]]
