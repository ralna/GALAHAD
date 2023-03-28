libgalahad_src += files('filtrane.F90')
if build_ciface
  libgalahad_c_src += files('C/filtrane_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usefiltrane.F90', 'runfiltrane_sif.F90')
endif
galahad_examples += [['filtranes', files('filtranes.f90')]]
