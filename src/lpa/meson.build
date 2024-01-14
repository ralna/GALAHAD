libgalahad_src += files('lpa.F90')

galahad_binaries += [['inlpa', files('inlpa.F90')]]

libgalahad_c_src += files('C/lpa_ciface.F90')

libgalahad_python_src += [['lpa', files('Python/lpa_pyiface.c',
                                        '../rpd/Python/rpd_pyiface.c','../lms/Python/lms_pyiface.c')]]

libgalahad_cutest_src += files('uselpa.F90')

galahad_cutest_binaries += [['runlpa_sif', files('runlpa_sif.F90')]]

galahad_tests += [['lpa', 'lpat', files('lpat.F90')],
                  ['lpa', 'lpati', files('lpati.F90')]]

galahad_c_tests += [['lpa', 'lpat_c', files('C/lpat.c')],
                    ['lpa', 'lpatf_c', files('C/lpatf.c')]]

galahad_python_tests += [['lpa', 'lpa_py', files('Python/test_lpa.py')]]

galahad_examples += [['lpas', files('lpas.f90')]]
