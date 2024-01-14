libgalahad_src += files('bgo.F90')

libgalahad_c_src += files('C/bgo_ciface.F90')

libgalahad_python_src += [['bgo', files('Python/bgo_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../trs/Python/trs_pyiface.c',
                                        '../gltr/Python/gltr_pyiface.c','../lhs/Python/lhs_pyiface.c','../lms/Python/lms_pyiface.c',
                                        '../psls/Python/psls_pyiface.c','../sha/Python/sha_pyiface.c','../trb/Python/trb_pyiface.c',
                                        '../ugo/Python/ugo_pyiface.c','../sils/Python/sils_pyiface.c','../rpd/Python/rpd_pyiface.c')]]

libgalahad_cutest_src += files('usebgo.F90')

galahad_cutest_binaries += [['runbgo_sif', files('runbgo_sif.F90')]]

galahad_tests += [['bgo', 'bgot', files('bgot.F90')],
                  ['bgo', 'bgoti', files('bgoti.F90')]]

galahad_c_tests += [['bgo', 'bgot_c', files('C/bgot.c')],
                    ['bgo', 'bgotf_c', files('C/bgotf.c')]]

galahad_python_tests += [['bgo', 'bgo_py', files('Python/test_bgo.py')],
                         ['bgo', 'bgo_quadratic_py', files('Python/test_bgo_quadratic.py')]]

galahad_examples += [['bgos', files('bgos.f90')],
                     ['bgos2', files('bgos2.f90')]]

galahad_c_examples += [['bgos_c', files('C/bgos.c')],
                       ['bgos2_c', files('C/bgos2.c')],
                       ['bgos3_c', files('C/bgos3.c')],
                       ['bgos4_c', files('C/bgos4.c')]]
