libgalahad_src += files('sls.F90')

libgalahad_c_src += files('C/sls_ciface.F90')

libgalahad_python_src += [['sls', files('Python/sls_pyiface.c',
                                        '../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usesls.F90')

galahad_cutest_binaries += [['runsls_sif', files('runsls_sif.F90')]]

galahad_tests += [['sls', 'slst', files('slst.F90')],
                  ['sls', 'slsti', files('slsti.F90')]]

galahad_c_tests += [['sls', 'slst_c', files('C/slst.c')],
                    ['sls', 'slstf_c', files('C/slstf.c')]]

galahad_python_tests += [['sls', 'sls_py', files('Python/test_sls.py')]]

galahad_examples += [['slss', files('slss.f90')]]
