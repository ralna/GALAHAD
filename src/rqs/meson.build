libgalahad_src += files('rqs.F90')
if build_ciface
  libgalahad_c_src += files('C/rqs_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('userqs.F90', 'runrqs_sif.F90')
endif
