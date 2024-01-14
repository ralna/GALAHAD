libgalahad_src += files('bqpb.F90')

galahad_binaries += [['inbqpb', files('inbqpb.F90')]]

libgalahad_c_src += files('C/bqpb_ciface.F90')

libgalahad_python_src += [['bqpb', files('Python/bqpb_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c','../sls/Python/sls_pyiface.c','../uls/Python/uls_pyiface.c',
                                         '../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c','../gls/Python/gls_pyiface.c',
                                         '../scu/Python/scu_pyiface.c','../fit/Python/fit_pyiface.c','../ir/Python/ir_pyiface.c',
                                         '../cro/Python/cro_pyiface.c','../lms/Python/lms_pyiface.c','../rpd/Python/rpd_pyiface.c',
                                         '../fdc/Python/fdc_pyiface.c')]]

libgalahad_cutest_src += files('usebqpb.F90')

galahad_cutest_binaries += [['runbqpb_sif', files('runbqpb_sif.F90')]]

galahad_tests += [['bqpb', 'bqpbt', files('bqpbt.F90')],
                  ['bqpb', 'bqpbti', files('bqpbti.F90')]]

galahad_c_tests += [['bqpb', 'bqpbt_c', files('C/bqpbt.c')],
                    ['bqpb', 'bqpbtf_c', files('C/bqpbtf.c')]]

galahad_python_tests += [['bqpb', 'bqpb_py', files('Python/test_bqpb.py')]]

galahad_examples += [['bqpbs', files('bqpbs.f90')],
                     ['bqpbs2', files('bqpbs2.f90')]]
