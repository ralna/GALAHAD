libgalahad_src += files('dlp.F90')
bingalahad_src += [['indlp', files('indlp.F90')]]
libgalahad_c_src += files('C/dlp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usedlp.F90', 'rundlp_sif.F90')
endif
