libgalahad_src += files('gltr.F90')

libgalahad_c_src += files('C/gltr_ciface.F90')

libgalahad_python_src += [['gltr', files('Python/gltr_pyiface.c')]]

libgalahad_cutest_src += files('usegltr.F90')

galahad_cutest_binaries += [['rungltr_sif', files('rungltr_sif.F90')]]

galahad_tests += [['gltr', 'gltrt', files('gltrt.F90')],
                  ['gltr', 'gltrti', files('gltrti.F90')]]

galahad_c_tests += [['gltr', 'gltrt_c', files('C/gltrt.c')]]

galahad_python_tests += [['gltr', 'gltr_py', files('Python/test_gltr.py')]]

galahad_examples += [['gltrs', files('gltrs.f90')],
                     ['gltrs2', files('gltrs2.f90')],
                     ['gltrs3', files('gltrs3.f90')]]
