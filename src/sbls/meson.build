libgalahad_src += files('sbls.F90')
if build_ciface
  libgalahad_c_src += files('C/sbls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usesbls.F90', 'runsbls_sif.F90')
endif
galahad_examples += [['sblss', files('sblss.f90')]]
