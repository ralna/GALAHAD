libgalahad_src += files('sha.F90')

libgalahad_c_src += files('C/sha_ciface.F90')

libgalahad_python_src += [['sha', files('Python/sha_pyiface.c')]]

libgalahad_cutest_src += files('usesha.F90')

galahad_cutest_binaries += [['runsha_sif', files('runsha_sif.F90')]]

galahad_tests += [['sha', 'shat', files('shat.F90')]]

galahad_python_tests += [['sha', 'sha_py', files('Python/test_sha.py')]]

galahad_examples += [['shas', files('shas.f90')]]
