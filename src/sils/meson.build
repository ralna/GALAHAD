libgalahad_src += files('sils.F90')
if build_ciface
  libgalahad_c_src += files('C/sils_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usesils.F90', 'runsils_sif.F90')
endif
galahad_examples += [['silss', files('silss.f90')]]
