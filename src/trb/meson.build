libgalahad_src += files('trb.F90')
if build_ciface
  libgalahad_c_src += files('C/trb_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usetrb.F90', 'runtrb_sif.F90')
endif
