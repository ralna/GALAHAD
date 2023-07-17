libgalahad_src += files('sec.F90')

if build_ciface
  libgalahad_c_src += files('C/sec_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['sec', files('Python/sec_pyiface.c')]]
endif

galahad_tests += [['sec', 'sect', files('sect.F90')]]

galahad_python_tests += [['sec', 'sec_py', files('Python/test_sec.py')]]

galahad_examples += [['secs', files('secs.f90')]]
