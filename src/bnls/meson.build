libgalahad_src += files('bnls.F90')

libgalahad_c_src += files('C/bnls_ciface.F90')

libgalahad_python_src += [['bnls', files('Python/bnls_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../gltr/Python/gltr_pyiface.c',
                                        '../glrt/Python/glrt_pyiface.c','../rqs/Python/rqs_pyiface.c','../psls/Python/psls_pyiface.c',
                                        '../bsc/Python/bsc_pyiface.c','../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c',)]]

libgalahad_cutest_src += files('usebnls.F90')

galahad_cutest_binaries += [['runbnls_sif', files('runbnls_sif.F90')]]

galahad_tests += [['bnls', 'bnlst', files('bnlst.F90')],
                  ['bnls', 'bnlst2', files('bnlst2.F90')],
                  ['bnls', 'bnlsti', files('bnlsti.F90')]]

galahad_c_tests += [['bnls', 'bnlst_c', files('C/bnlst.c')],
                    ['bnls', 'bnlstf_c', files('C/bnlstf.c')]]

galahad_python_tests += [['bnls', 'bnls_py', files('Python/test_bnls.py')]]

galahad_examples += [['bnlss', files('bnlss.f90')],
                     ['bnlss2', files('bnlss2.f90')],
                     ['bnlss3', files('bnlss3.f90')]]

galahad_c_examples += [['bnlss_c', files('C/bnlss.c')],
                       ['bnlss2_c', files('C/bnlss2.c')],
                       ['bnlss3_c', files('C/bnlss3.c')],
                       ['bnlss4_c', files('C/bnlss4.c')]]
