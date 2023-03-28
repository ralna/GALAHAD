libgalahad_src += files('tru.F90')
if build_ciface
  libgalahad_c_src += files('C/tru_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usetru.F90', 'runtru_sif.F90')
endif
galahad_c_examples += [['trus_c', files('C/trus.c')],
                       ['trus2_c', files('C/trus2.c')],
                       ['trus3_c', files('C/trus3.c')],
                       ['trus4_c', files('C/trus4.c')]]
