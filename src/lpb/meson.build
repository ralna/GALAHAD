libgalahad_src += files('lpb.F90')
bingalahad_src += [['inlpb', files('inlpb.F90')]]
if build_ciface
  libgalahad_c_src += files('C/lpb_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('uselpb.F90', 'runlpb_sif.F90')
endif
galahad_examples += [['lpbs', files('lpbs.f90')]]
