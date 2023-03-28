libgalahad_src += files('eqp.F90')
if build_ciface
  libgalahad_c_src += files('C/eqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useeqp.F90', 'runeqp_sif.F90')
endif
galahad_examples += [['eqps', files('eqps.f90')]]
