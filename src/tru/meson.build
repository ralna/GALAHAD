libgalahad_src += files('tru.F90')

libgalahad_c_src += files('C/tru_ciface.F90')

libgalahad_python_src += [['tru', files('Python/tru_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../trs/Python/trs_pyiface.c',
                                        '../gltr/Python/gltr_pyiface.c','../psls/Python/psls_pyiface.c','../dps/Python/dps_pyiface.c',
                                        '../lms/Python/lms_pyiface.c','../sec/Python/sec_pyiface.c','../sha/Python/sha_pyiface.c',
                                        '../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usetru.F90')

galahad_cutest_binaries += [['runtru_sif', files('runtru_sif.F90')]]

galahad_tests += [['tru', 'trut', files('trut.F90')],
                  ['tru', 'truti', files('truti.F90')]]

galahad_c_tests += [['tru', 'trut_c', files('C/trut.c')],
                    ['tru', 'trutf_c', files('C/trutf.c')]]

galahad_python_tests += [['tru', 'tru_py', files('Python/test_tru.py')]]

galahad_examples += [['trus', files('trus.f90')],
                     ['trus2', files('trus2.f90')],
                     ['trus3', files('trus3.f90')],
                     ['trus4', files('trus4.f90')]]

galahad_c_examples += [['trus_c', files('C/trus.c')],
                       ['trus2_c', files('C/trus2.c')],
                       ['trus3_c', files('C/trus3.c')],
                       ['trus4_c', files('C/trus4.c')]]
