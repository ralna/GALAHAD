libgalahad_src += files('lqr.F90')
if libcutest.found()
  libgalahad_cutest_src += files('uselqr.F90', 'runlqr_sif.F90')
endif
galahad_examples += [['lqrs', files('lqrs.f90')],
                     ['lqrs2', files('lqrs2.f90')]]
