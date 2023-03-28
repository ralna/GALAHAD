libgalahad_src += files('qpb.F90')
bingalahad_src += [['inqpb', files('inqpb.F90')]]
libgalahad_c_src += files('C/qpb_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('useqpb.F90', 'runqpb_sif.F90')
endif
