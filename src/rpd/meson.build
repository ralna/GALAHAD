libgalahad_src += files('rpd.F90')

libgalahad_c_src += files('C/rpd_ciface.F90')

libgalahad_python_src += [['rpd', files('Python/rpd_pyiface.c')]]

galahad_tests += [['rpd', 'rpdt', files('rpdt.F90')],
                  ['rpd', 'rpdti', files('rpdti.F90')]]

galahad_c_tests += [['rpd', 'rpdt_c', files('C/rpdt.c')],
                    ['rpd', 'rpdtf_c', files('C/rpdtf.c')]]

galahad_python_tests += [['rpd', 'rpd_py', files('Python/test_rpd.py')]]

galahad_examples += [['rpds', files('rpds.f90')]]
