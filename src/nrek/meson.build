libgalahad_src += files('nrek.F90')

libgalahad_c_src += files('C/nrek_ciface.F90')

libgalahad_python_src += [['nrek', files('Python/nrek_pyiface.c',
                                        '../nodend/Python/nodend_pyiface.c',
                                        '../sls/Python/sls_pyiface.c',
                                        '../trs/Python/trs_pyiface.c')]]

libgalahad_cutest_src += files('usenrek.F90')

galahad_cutest_binaries += [['runnrek_sif', files('runnrek_sif.F90')]]

galahad_tests += [['nrek', 'nrekt', files('nrekt.F90')],
                  ['nrek', 'nrekti', files('nrekti.F90')]]

galahad_c_tests += [['nrek', 'nrekt_c', files('C/nrekt.c')],
                    ['nrek', 'nrektf_c', files('C/nrektf.c')]]

galahad_python_tests += [['nrek', 'nrek_py', files('Python/test_nrek.py')]]

galahad_examples += [['nreks', files('nreks.f90')],
                     ['nreks2', files('nreks2.f90')]]
