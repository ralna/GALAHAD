libgalahad_src += files('rqs.F90')

libgalahad_c_src += files('C/rqs_ciface.F90')

libgalahad_python_src += [['rqs', files('Python/rqs_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('userqs.F90')

galahad_cutest_binaries += [['runrqs_sif', files('runrqs_sif.F90')]]

galahad_tests += [['rqs', 'rqst', files('rqst.F90')],
                  ['rqs', 'rqsti', files('rqsti.F90')]]

galahad_c_tests += [['rqs', 'rqst_c', files('C/rqst.c')],
                    ['rqs', 'rqstf_c', files('C/rqstf.c')]]

galahad_python_tests += [['rqs', 'rqs_py', files('Python/test_rqs.py')]]

galahad_examples += [['rqss', files('rqss.f90')],
                     ['rqss2', files('rqss2.f90')]]
