libgalahad_src += files('wcp.F90')

libgalahad_c_src += files('C/wcp_ciface.F90')

libgalahad_python_src += [['wcp', files('Python/wcp_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c','../fdc/Python/fdc_pyiface.c','../sls/Python/sls_pyiface.c',
                                        '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c','../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usewcp.F90')

galahad_cutest_binaries += [['runwcp_sif', files('runwcp_sif.F90')]]

galahad_tests += [['wcp', 'wcpt', files('wcpt.F90')],
                  ['wcp', 'wcpti', files('wcpti.F90')]]

galahad_c_tests += [['wcp', 'wcpt_c', files('C/wcpt.c')],
                    ['wcp', 'wcptf_c', files('C/wcptf.c')]]

galahad_python_tests += [['wcp', 'wcp_py', files('Python/test_wcp.py')]]

galahad_examples += [['wcps', files('wcps.f90')]]
