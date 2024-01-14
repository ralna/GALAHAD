libgalahad_src += files('fdc.F90')

libgalahad_c_src += files('C/fdc_ciface.F90')

libgalahad_python_src += [['fdc', files('Python/fdc_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../uls/Python/uls_pyiface.c','../sils/Python/sils_pyiface.c',
                                        '../gls/Python/gls_pyiface.c')]]

galahad_tests += [['fdc', 'fdct', files('fdct.F90')],
                  ['fdc', 'fdcti', files('fdcti.F90')]]

galahad_c_tests += [['fdc', 'fdct_c', files('C/fdct.c')],
                    ['fdc', 'fdctf_c', files('C/fdctf.c')]]

galahad_python_tests += [['fdc', 'fdc_py', files('Python/test_fdc.py')]]

galahad_examples += [['fdcs', files('fdcs.f90')]]
