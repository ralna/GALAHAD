libgalahad_src += files('cro.F90')

libgalahad_c_src += files('C/cro_ciface.F90')

libgalahad_python_src += [['cro', files('Python/cro_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c','../lms/Python/lms_pyiface.c','../scu/Python/scu_pyiface.c',
                                        '../ir/Python/ir_pyiface.c','../rpd/Python/rpd_pyiface.c','../sls/Python/sls_pyiface.c',
                                        '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c')]]

galahad_tests += [['cro', 'crot', files('crot.F90')],
                  ['cro', 'croti', files('croti.F90')]]

galahad_c_tests += [['cro', 'crot_c', files('C/crot.c')],
                    ['cro', 'crotf_c', files('C/crotf.c')]]

galahad_python_tests += [['cro', 'cro_py', files('Python/test_cro.py')]]

galahad_examples += [['cros', files('cros.f90')],
                     ['cros2', files('cros2.f90')],
                     ['cros3', files('cros3.f90')]]
