libgalahad_src += files('lstr.F90')

libgalahad_c_src += files('C/lstr_ciface.F90')

libgalahad_python_src += [['lstr', files('Python/lstr_pyiface.c')]]

libgalahad_cutest_src += files('uselstr.F90')

galahad_cutest_binaries += [['runlstr_sif', files('runlstr_sif.F90')]]

galahad_tests += [['lstr', 'lstrt', files('lstrt.F90')],
                  ['lstr', 'lstrti', files('lstrti.F90')]]

galahad_c_tests += [['lstr', 'lstrt_c', files('C/lstrt.c')]]

galahad_python_tests += [['lstr', 'lstr_py', files('Python/test_lstr.py')]]

galahad_examples += [['lstrs', files('lstrs.f90')],
                     ['lstrs2', files('lstrs2.f90')]]
