libgalahad_src += files('l2rt.F90')
if build_ciface
  libgalahad_c_src += files('C/l2rt_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usel2rt.F90', 'runl2rt_sif.F90')
endif
