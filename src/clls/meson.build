libgalahad_src += files('clls.F90')

libgalahad_c_src += files('C/clls_ciface.F90')

libgalahad_python_src += [['clls', files('Python/clls_pyiface.c',
                                         '../fdc/Python/fdc_pyiface.c','../sls/Python/sls_pyiface.c',
                                         '../fit/Python/fit_pyiface.c','../roots/Python/roots_pyiface.c',
                                         '../cro/Python/cro_pyiface.c','../rpd/Python/rpd_pyiface.c',
                                         '../ir/Python/ir_pyiface.c','../uls/Python/uls_pyiface.c',
                                         '../sils/Python/sils_pyiface.c','../sbls/Python/sbls_pyiface.c',
                                         '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('useclls.F90')

galahad_cutest_binaries += [['runclls_sif', files('runclls_sif.F90')]]

galahad_tests += [['clls', 'cllst', files('cllst.F90')],
                  ['clls', 'cllsti', files('cllsti.F90')]]

galahad_c_tests += [['clls', 'cllst_c', files('C/cllst.c')],
                    ['clls', 'cllstf_c', files('C/cllstf.c')]]

galahad_python_tests += [['clls', 'clls_py', files('Python/test_clls.py')]]

galahad_examples += [['cllss', files('cllss.f90')],
                     ['cllss2', files('cllss2.f90')],
                     ['cllss3', files('cllss3.f90')]]
