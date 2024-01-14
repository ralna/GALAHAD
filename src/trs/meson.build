libgalahad_src += files('trs.F90')

libgalahad_c_src += files('C/trs_ciface.F90')

libgalahad_python_src += [['trs', files('Python/trs_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c',
                                        '../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usetrs.F90')

galahad_cutest_binaries += [['runtrs_sif', files('runtrs_sif.F90')]]

galahad_tests += [['trs', 'trst', files('trst.F90')],
                  ['trs', 'trsti', files('trsti.F90')]]

galahad_c_tests += [['trs', 'trst_c', files('C/trst.c')],
                    ['trs', 'trstf_c', files('C/trstf.c')]]

galahad_python_tests += [['trs', 'trs_py', files('Python/test_trs.py')]]

galahad_examples += [['trss', files('trss.f90')],
                     ['trss2', files('trss2.f90')]]
