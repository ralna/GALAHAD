libgalahad_src += files('nls.F90')

libgalahad_c_src += files('C/nls_ciface.F90')

libgalahad_python_src += [['nls', files('Python/nls_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../gltr/Python/gltr_pyiface.c',
                                        '../glrt/Python/glrt_pyiface.c','../rqs/Python/rqs_pyiface.c','../psls/Python/psls_pyiface.c',
                                        '../bsc/Python/bsc_pyiface.c','../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c',)]]

libgalahad_cutest_src += files('usenls.F90')

galahad_cutest_binaries += [['runnls_sif', files('runnls_sif.F90')]]

galahad_tests += [['nls', 'nlst', files('nlst.F90')],
                  ['nls', 'nlsti', files('nlsti.F90')]]

galahad_c_tests += [['nls', 'nlst_c', files('C/nlst.c')],
                    ['nls', 'nlstf_c', files('C/nlstf.c')]]

galahad_python_tests += [['nls', 'nls_py', files('Python/test_nls.py')]]

galahad_examples += [['nlss', files('nlss.f90')],
                     ['nlss2', files('nlss2.f90')],
                     ['nlss3', files('nlss3.f90')]]

galahad_c_examples += [['nlss_c', files('C/nlss.c')],
                       ['nlss2_c', files('C/nlss2.c')],
                       ['nlss3_c', files('C/nlss3.c')],
                       ['nlss4_c', files('C/nlss4.c')]]
