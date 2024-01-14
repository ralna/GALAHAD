libgalahad_src += files('qpb.F90')

galahad_binaries += [['inqpb', files('inqpb.F90')]]

libgalahad_c_src += files('C/qpb_ciface.F90')

libgalahad_python_src += [['qpb', files('Python/qpb_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../sbls/Python/sbls_pyiface.c','../fit/Python/fit_pyiface.c',
                                        '../gltr/Python/gltr_pyiface.c','../fdc/Python/fdc_pyiface.c','../lsqp/Python/lsqp_pyiface.c',
                                        '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c',)]]

libgalahad_cutest_src += files('useqpb.F90')

galahad_cutest_binaries += [['runqpb_sif', files('runqpb_sif.F90')]]

galahad_tests += [['qpb', 'qpbt', files('qpbt.F90')],
                  ['qpb', 'qpbti', files('qpbti.F90')]]

galahad_c_tests += [['qpb', 'qpbt_c', files('C/qpbt.c')],
                    ['qpb', 'qpbtf_c', files('C/qpbtf.c')]]

galahad_python_tests += [['qpb', 'qpb_py', files('Python/test_qpb.py')]]

galahad_examples += [['qpbs', files('qpbs.f90')],
                     ['qpbs2', files('qpbs2.f90')]]
