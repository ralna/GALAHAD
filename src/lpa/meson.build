libgalahad_src += files('lpa.F90')

galahad_binaries += [['inlpa', files('inlpa.F90')]]

if build_ciface
  libgalahad_c_src += files('C/lpa_ciface.F90')
endif

if libcutest.found()
  libgalahad_cutest_src += files('uselpa.F90', 'runlpa_sif.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/lpa_pyiface.c')
endif

galahad_tests += [['lpa', 'lpat', files('lpat.F90')],
                  ['lpa', 'lpati', files('lpati.F90')]]

galahad_c_tests += [['lpa', 'lpat_c', files('C/lpat.c')],
                    ['lpa', 'lpatf_c', files('C/lpatf.c')]]

galahad_python_tests += [['lpa', 'lpa_py', files('Python/test_lpa.py')]]

galahad_examples += [['lpas', files('lpas.f90')]]
