libgalahad_src += files('roots.F90')

if build_ciface
  libgalahad_c_src += files('C/roots_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['roots', files('Python/roots_pyiface.c')]]
endif

galahad_tests += [['roots', 'rootst', files('rootst.F90')]]

galahad_python_tests += [['roots', 'roots_py', files('Python/test_roots.py')]]

galahad_examples += [['rootss', files('rootss.f90')]]
