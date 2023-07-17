libgalahad_src += files('bsc.F90')

if build_ciface
  libgalahad_c_src += files('C/bsc_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['bsc', files('Python/bsc_pyiface.c')]]
endif

galahad_tests += [['bsc', 'bsct', files('bsct.F90')]]

galahad_python_tests += [['bsc', 'bsc_py', files('Python/test_bsc.py')]]

galahad_examples += [['bscs', files('bscs.f90')]]
