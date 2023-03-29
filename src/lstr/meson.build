libgalahad_src += files('lstr.F90')
if build_ciface
  libgalahad_c_src += files('C/lstr_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('uselstr.F90', 'runlstr_sif.F90')
endif

galahad_tests += [['lstr', 'lstrt', files('lstrt.F90')],
                  ['lstr', 'lstrti', files('lstrti.F90')]]

galahad_c_tests += [['lstr', 'lstrt_c', files('C/lstrt.c')]]

galahad_examples += [['lstrs', files('lstrs.f90')],
                     ['lstrs2', files('lstrs2.f90')]]
