libgalahad_src += files('sils.F90')

if build_ciface
  libgalahad_c_src += files('C/sils_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/sils_pyiface.c')
endif

if libcutest.found()
  libgalahad_cutest_src += files('usesils.F90', 'runsils_sif.F90')
endif

galahad_tests += [['sils', 'silst', files('silst.F90')]]

galahad_python_tests += [['sils', 'sils_py', files('Python/test_sils.py')]]

galahad_examples += [['silss', files('silss.f90')]]
