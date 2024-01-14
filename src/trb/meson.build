libgalahad_src += files('trb.F90')

libgalahad_c_src += files('C/trb_ciface.F90')

libgalahad_python_src += [['trb', files('Python/trb_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../trs/Python/trs_pyiface.c',
                                        '../gltr/Python/gltr_pyiface.c','../psls/Python/psls_pyiface.c','../lms/Python/lms_pyiface.c',
                                        '../sha/Python/sha_pyiface.c','../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usetrb.F90')

galahad_cutest_binaries += [['runtrb_sif', files('runtrb_sif.F90')]]

galahad_tests += [['trb', 'trbt', files('trbt.F90')],
                  ['trb', 'trbti', files('trbti.F90')]]

galahad_c_tests += [['trb', 'trbt_c', files('C/trbt.c')],
                    ['trb', 'trbtf_c', files('C/trbtf.c')]]

galahad_python_tests += [['trb', 'trb_py', files('Python/test_trb.py')]]

galahad_examples += [['trbs', files('trbs.f90')],
                     ['trbs2', files('trbs2.f90')],
                     ['trbs3', files('trbs3.f90')]]

galahad_c_examples += [['trbs_c', files('C/trbs.c')],
                       ['trbs2_c', files('C/trbs2.c')],
                       ['trbs3_c', files('C/trbs3.c')],
                       ['trbs4_c', files('C/trbs4.c')]]
