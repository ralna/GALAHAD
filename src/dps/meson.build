libgalahad_src += files('dps.F90')
if build_ciface
  libgalahad_c_src += files('C/dps_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usedps.F90', 'rundps_sif.F90')
endif
galahad_examples += [['dpss', files('dpss.f90')],
                     ['dpss2', files('dpss2.f90')]]
