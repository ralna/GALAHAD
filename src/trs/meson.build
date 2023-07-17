libgalahad_src += files('trs.F90')

if build_ciface
  libgalahad_c_src += files('C/trs_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['trs', files('Python/trs_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('usetrs.F90', 'runtrs_sif.F90')
endif

galahad_tests += [['trs', 'trst', files('trst.F90')],
                  ['trs', 'trsti', files('trsti.F90')]]

galahad_c_tests += [['trs', 'trst_c', files('C/trst.c')],
                    ['trs', 'trstf_c', files('C/trstf.c')]]

galahad_python_tests += [['trs', 'trs_py', files('Python/test_trs.py')]]

galahad_examples += [['trss', files('trss.f90')],
                     ['trss2', files('trss2.f90')]]
