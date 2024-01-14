libgalahad_src += files('sbls.F90')

libgalahad_c_src += files('C/sbls_ciface.F90')

libgalahad_python_src += [['sbls', files('Python/sbls_pyiface.c',
                                         '../sls/Python/sls_pyiface.c','../sils/Python/sils_pyiface.c',
                                         '../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usesbls.F90')

galahad_cutest_binaries += [['runsbls_sif', files('runsbls_sif.F90')]]

galahad_tests += [['sbls', 'sblst', files('sblst.F90')],
                  ['sbls', 'sblsti', files('sblsti.F90')]]

galahad_c_tests += [['sbls', 'sblst_c', files('C/sblst.c')],
                    ['sbls', 'sblstf_c', files('C/sblstf.c')]]

galahad_python_tests += [['sbls', 'sbls_py', files('Python/test_sbls.py')]]

galahad_examples += [['sblss', files('sblss.f90')]]
