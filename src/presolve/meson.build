libgalahad_src += files('presolve.F90')

libgalahad_c_src += files('C/presolve_ciface.F90')

libgalahad_python_src += [['presolve', files('Python/presolve_pyiface.c')]]

libgalahad_cutest_src += files('usepresolve.F90')

galahad_cutest_binaries += [['runpresolve_sif', files('runpresolve_sif.F90')]]

galahad_tests += [['presolve', 'presolvet', files('presolvet.F90')],
                  ['presolve', 'presolveti', files('presolveti.F90')]]

galahad_c_tests += [['presolve', 'presolvet_c', files('C/presolvet.c')],
                    ['presolve', 'presolvetf_c', files('C/presolvetf.c')]]

galahad_python_tests += [['presolve', 'presolve_py', files('Python/test_presolve.py')]]

galahad_examples += [['presolves', files('presolves.f90')]]
