libgalahad_src += files('arc.F90')
if build_ciface
  libgalahad_c_src += files('C/arc_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usearc.F90', 'runarc_sif.F90')
endif
