libgalahad_src += files('presolve.F90')
if build_ciface
  libgalahad_c_src += files('C/presolve_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usepresolve.F90', 'runpresolve_sif.F90')
endif
