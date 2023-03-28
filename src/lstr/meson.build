libgalahad_src += files('lstr.F90')
if build_ciface
  libgalahad_c_src += files('C/lstr_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('uselstr.F90', 'runlstr_sif.F90')
endif
