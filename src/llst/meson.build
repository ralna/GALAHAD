libgalahad_src += files('llst.F90')

libgalahad_c_src += files('C/llst_ciface.F90')

libgalahad_python_src += [['llst', files('Python/llst_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c','../sls/Python/sls_pyiface.c','../uls/Python/uls_pyiface.c',
                                         '../roots/Python/roots_pyiface.c','../sils/Python/sils_pyiface.c','../gls/Python/gls_pyiface.c',
                                         '../ir/Python/ir_pyiface.c')]]

galahad_tests += [['llst', 'llstt', files('llstt.F90')],
                  ['llst', 'llstti', files('llstti.F90')]]

galahad_c_tests += [['llst', 'llstt_c', files('C/llstt.c')],
                    ['llst', 'llsttf_c', files('C/llsttf.c')]]

galahad_python_tests += [['llst', 'llst_py', files('Python/test_llst.py')]]

galahad_examples += [['llsts', files('llsts.f90')],
                     ['llsts2', files('llsts2.f90')],
                     ['llsts3', files('llsts3.f90')]]
