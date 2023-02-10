libgalahad_src += files('uls.F90')

if build_ciface
  libgalahad_c_src += files('C/uls_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/uls_pyiface.c')
endif

galahad_tests += [['uls', 'ulst', files('ulst.F90')],
                  ['uls', 'ulsti', files('ulsti.F90')]]

galahad_c_tests += [['uls', 'ulst_c', files('C/ulst.c')],
                    ['uls', 'ulstf_c', files('C/ulstf.c')]]

galahad_examples += [['ulss', files('ulss.f90')]]
