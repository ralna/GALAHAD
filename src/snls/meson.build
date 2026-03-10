libgalahad_src += files('snls.F90')

libgalahad_c_src += files('C/snls_ciface.F90')

libgalahad_python_src += [['snls', files('Python/snls_pyiface.c',
                                        '../nodend/Python/nodend_pyiface.c',
                                        '../sls/Python/sls_pyiface.c',
                                        '../ir/Python/ir_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c',
                                        '../clls/Python/clls_pyiface.c',
                                        '../slls/Python/slls_pyiface.c',
                                        '../sllsb/Python/sllsb_pyiface.c',
                                        '../psls/Python/psls_pyiface.c',
                                        '../fdc/Python/fdc_pyiface.c',
                                        '../bsc/Python/bsc_pyiface.c',
                                        '../roots/Python/roots_pyiface.c',
                                        '../sils/Python/sils_pyiface.c',)]]

libgalahad_cutest_src += files('usesnls.F90')

galahad_cutest_binaries += [['runsnls_sif', files('runsnls_sif.F90')]]

galahad_tests += [['snls', 'snlst', files('snlst.F90')],
                  ['snls', 'snlsti', files('snlsti.F90')]]

galahad_c_tests += [['snls', 'snlst_c', files('C/snlst.c')],
                    ['snls', 'snlstf_c', files('C/snlstf.c')]]

galahad_python_tests += [['snls', 'snls_py', files('Python/test_snls.py')]]

galahad_examples += [['snlss', files('snlss.f90')],
                     ['snlss2', files('snlss2.f90')],
                     ['snlss3', files('snlss3.f90')],
                     ['snlss4', files('snlss4.f90')]]

galahad_c_examples += [['snlss_c', files('C/snlss.c')],
                       ['snlss2_c', files('C/snlss2.c')],
                       ['snlss3_c', files('C/snlss3.c')],
                       ['snlss4_c', files('C/snlss4.c')]]


