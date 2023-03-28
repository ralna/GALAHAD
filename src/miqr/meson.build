libgalahad_src += files('miqr.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usemiqr.F90', 'runmiqr_sif.F90')
endif
galahad_examples += [['miqrs', files('miqrs.f90')],
                     ['miqrs2', files('miqrs2.f90')]]
