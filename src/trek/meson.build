libgalahad_src += files('trek.F90')

libgalahad_c_src += files('C/trek_ciface.F90')

libgalahad_python_src += [['trek', files('Python/trek_pyiface.c',
                                        '../nodend/Python/nodend_pyiface.c',
                                        '../sls/Python/sls_pyiface.c',
                                        '../trs/Python/trs_pyiface.c')]]

libgalahad_cutest_src += files('usetrek.F90')

galahad_cutest_binaries += [['runtrek_sif', files('runtrek_sif.F90')]]

galahad_tests += [['trek', 'trekt', files('trekt.F90')],
                  ['trek', 'trekti', files('trekti.F90')]]

galahad_c_tests += [['trek', 'trekt_c', files('C/trekt.c')],
                    ['trek', 'trektf_c', files('C/trektf.c')]]

galahad_python_tests += [['trek', 'trek_py', files('Python/test_trek.py')]]

galahad_examples += [['treks', files('treks.f90')],
                     ['treks2', files('treks2.f90')]]
