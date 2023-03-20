libgalahad_src += files('sha.F90')
libgalahad_c_src += files('C/sha_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usesha.F90', 'runsha_sif.F90')
endif
