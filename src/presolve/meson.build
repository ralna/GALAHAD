libgalahad_src += files('presolve.F90')
if build_ciface
  libgalahad_c_src += files('C/presolve_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usepresolve.F90', 'runpresolve_sif.F90')
endif

galahad_c_tests += [['presolve', 'presolvet_c', files('C/presolvet.c')],
                    ['presolve', 'presolvetf_c', files('C/presolvetf.c')]]

galahad_examples += [['presolves', files('presolves.f90')]]
