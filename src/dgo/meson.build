libgalahad_src += files('dgo.F90')

libgalahad_c_src += files('C/dgo_ciface.F90')

libgalahad_python_src += [['dgo', files('Python/dgo_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../trs/Python/trs_pyiface.c',
                                        '../gltr/Python/gltr_pyiface.c','../lhs/Python/lhs_pyiface.c','../lms/Python/lms_pyiface.c',
                                        '../psls/Python/psls_pyiface.c','../sha/Python/sha_pyiface.c','../trb/Python/trb_pyiface.c',
                                        '../ugo/Python/ugo_pyiface.c','../hash/Python/hash_pyiface.c','../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usedgo.F90')

galahad_cutest_binaries += [['rundgo_sif', files('rundgo_sif.F90')]]

galahad_tests += [['dgo', 'dgot', files('dgot.F90')],
                  ['dgo', 'dgoti', files('dgoti.F90')]]

galahad_c_tests += [['dgo', 'dgot_c', files('C/dgot.c')],
                    ['dgo', 'dgotf_c', files('C/dgotf.c')]]

galahad_python_tests += [['dgo', 'dgo_py', files('Python/test_dgo.py')],
                         ['dgo', 'dgo_quadratic_py', files('Python/test_dgo_quadratic.py')]]

galahad_examples += [['dgos', files('dgos.f90')],
                     ['dgos2', files('dgos2.f90')]]

galahad_c_examples += [['dgos_c', files('C/dgos.c')],
                       ['dgos2_c', files('C/dgos2.c')],
                       ['dgos3_c', files('C/dgos3.c')],
                       ['dgos4_c', files('C/dgos4.c')]]
