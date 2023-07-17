libgalahad_src += files('scu.F90')

if build_ciface
  libgalahad_c_src += files('C/scu_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['scu', files('Python/scu_pyiface.c')]]
endif

galahad_tests += [['scu', 'scut', files('scut.F90')]]

galahad_python_tests += [['scu', 'scu_py', files('Python/test_scu.py')]]

galahad_examples += [['scus', files('scus.f90')]]
