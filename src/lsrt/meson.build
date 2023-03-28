libgalahad_src += files('lsrt.F90')
if build_ciface
  libgalahad_c_src += files('C/lsrt_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('uselsrt.F90', 'runlsrt_sif.F90')
endif
