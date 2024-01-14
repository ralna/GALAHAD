libgalahad_src += files('glrt.F90')

libgalahad_c_src += files('C/glrt_ciface.F90')

libgalahad_python_src += [['glrt', files('Python/glrt_pyiface.c')]]

libgalahad_cutest_src += files('useglrt.F90')

galahad_cutest_binaries += [['runglrt_sif', files('runglrt_sif.F90')]]

galahad_tests += [['glrt', 'glrtt', files('glrtt.F90')]]

galahad_c_tests += [['glrt', 'glrtt_c', files('C/glrtt.c')]]

galahad_python_tests += [['glrt', 'glrt_py', files('Python/test_glrt.py')]]

galahad_examples += [['glrts', files('glrts.f90')]]
