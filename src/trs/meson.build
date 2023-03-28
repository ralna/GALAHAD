libgalahad_src += files('trs.F90')
if build_ciface
  libgalahad_c_src += files('C/trs_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usetrs.F90', 'runtrs_sif.F90')
endif
galahad_examples += [['trss', files('trss.f90')],
                     ['trss2', files('trss2.f90')]]
