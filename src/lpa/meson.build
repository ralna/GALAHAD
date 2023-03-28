libgalahad_src += files('lpa.F90')
bingalahad_src += [['inlpa', files('inlpa.F90')]]
if build_ciface
  libgalahad_c_src += files('C/lpa_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('uselpa.F90', 'runlpa_sif.F90')
endif
