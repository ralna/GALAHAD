libgalahad_src += files('slls.F90')
if build_ciface
  libgalahad_c_src += files('C/slls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useslls.F90', 'runslls_sif.F90')
endif
