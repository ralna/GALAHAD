libgalahad_src += files('dlp.F90')
galahad_binaries += [['indlp', files('indlp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/dlp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usedlp.F90', 'rundlp_sif.F90')
endif
galahad_examples += [['dlps', files('dlps.f90')]]
