libgalahad_src += files('sha.F90')
if build_ciface
  libgalahad_c_src += files('C/sha_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usesha.F90', 'runsha_sif.F90')
endif
galahad_examples += [['shas', files('shas.f90')]]
