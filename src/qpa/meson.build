libgalahad_src += files('qpa.F90')

galahad_binaries += [['inqpa', files('inqpa.F90')]]

libgalahad_c_src += files('C/qpa_ciface.F90')

libgalahad_python_src += [['qpa', files('Python/qpa_pyiface.c',
                                        '../sbls/Python/sbls_pyiface.c','../gltr/Python/gltr_pyiface.c','../sls/Python/sls_pyiface.c',
                                        '../sils/Python/sils_pyiface.c','../uls/Python/uls_pyiface.c', '../gls/Python/gls_pyiface.c',)]]

libgalahad_cutest_src += files('useqpa.F90')

galahad_cutest_binaries += [['runqpa_sif', files('runqpa_sif.F90')]]

galahad_tests += [['qpa', 'qpat', files('qpat.F90')],
                  ['qpa', 'qpati', files('qpati.F90')]]

galahad_c_tests += [['qpa', 'qpat_c', files('C/qpat.c')],
                    ['qpa', 'qpatf_c', files('C/qpatf.c')]]

galahad_python_tests += [['qpa', 'qpa_py', files('Python/test_qpa.py')]]

galahad_examples += [['qpas', files('qpas.f90')]]
