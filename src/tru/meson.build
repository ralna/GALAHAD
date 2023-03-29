libgalahad_src += files('tru.F90')
if build_ciface
  libgalahad_c_src += files('C/tru_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usetru.F90', 'runtru_sif.F90')
endif

galahad_tests += [['tru', 'trut', files('trut.F90')],
                  ['tru', 'truti', files('truti.F90')]]

galahad_c_tests += [['tru', 'trut_c', files('C/trut.c')],
                    ['tru', 'trutf_c', files('C/trutf.c')]]

galahad_examples += [['trus', files('trus.f90')],
                     ['trus2', files('trus2.f90')],
                     ['trus3', files('trus3.f90')],
                     ['trus4', files('trus4.f90')]]

galahad_c_examples += [['trus_c', files('C/trus.c')],
                       ['trus2_c', files('C/trus2.c')],
                       ['trus3_c', files('C/trus3.c')],
                       ['trus4_c', files('C/trus4.c')]]
