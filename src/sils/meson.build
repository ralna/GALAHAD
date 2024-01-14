libgalahad_src += files('sils.F90')

libgalahad_c_src += files('C/sils_ciface.F90')

libgalahad_python_src += [['sils', files('Python/sils_pyiface.c')]]

libgalahad_cutest_src += files('usesils.F90')

galahad_cutest_binaries += [['runsils_sif', files('runsils_sif.F90')]]

galahad_tests += [['sils', 'silst', files('silst.F90')]]

galahad_python_tests += [['sils', 'sils_py', files('Python/test_sils.py')]]

galahad_examples += [['silss', files('silss.f90')]]
