libgalahad_src += files('sls.F90')
if build_ciface
  libgalahad_c_src += files('C/sls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usesls.F90', 'runsls_sif.F90')
endif

galahad_c_tests += [['sls', 'slst_c', files('C/slst.c')],
                    ['sls', 'slstf_c', files('C/slstf.c')]]

galahad_examples += [['slss', files('slss.f90')]]
