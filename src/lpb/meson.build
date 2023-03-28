libgalahad_src += files('lpb.F90')
bingalahad_src += [['inlpb', files('inlpb.F90')]]
libgalahad_c_src += files('C/lpb_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('uselpb.F90', 'runlpb_sif.F90')
endif
