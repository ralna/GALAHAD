libgalahad_src += files('dps.F90')

libgalahad_c_src += files('C/dps_ciface.F90')

libgalahad_python_src += [['dps', files('Python/dps_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../sils/Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usedps.F90')

galahad_cutest_binaries += [['rundps_sif', files('rundps_sif.F90')]]

galahad_tests += [['dps', 'dpst', files('dpst.F90')]]

galahad_c_tests += [['dps', 'dpst_c', files('C/dpst.c')],
                    ['dps', 'dpstf_c', files('C/dpstf.c')]]

galahad_python_tests += [['dps', 'dps_py', files('Python/test_dps.py')]]

galahad_examples += [['dpss', files('dpss.f90')],
                     ['dpss2', files('dpss2.f90')]]
