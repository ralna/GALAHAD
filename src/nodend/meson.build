subdir('metis51')
subdir('metis52')

libgalahad_src += files('nodend.F90')

libgalahad_c_src += files('C/nodend_ciface.F90')

libgalahad_cc_src += files('adapter_nodend.c')

libgalahad_python_src += [['nodend', files('Python/nodend_pyiface.c')]]

libgalahad_cutest_src += files('usenodend.F90')

galahad_cutest_binaries += [['runnodend_sif', files('runnodend_sif.F90')]]

galahad_tests += [['nodend', 'nodendt', files('nodendt.F90')]]

galahad_c_tests += [['nodend', 'nodendt_c', files('C/nodendt.c')],
                    ['nodend', 'nodendtf_c', files('C/nodendtf.c')]]

galahad_python_tests += [['nodend', 'nodend_py', 
                          files('Python/test_nodend.py')]]

galahad_examples += [['nodends', files('nodends.f90')],
                     ['nodends2', files('nodends2.f90')]]
