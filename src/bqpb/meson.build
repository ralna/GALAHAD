libgalahad_src += files('bqpb.F90')
galahad_binaries += [['inbqpb', files('inbqpb.F90')]]
if build_ciface
  libgalahad_c_src += files('C/bqpb_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usebqpb.F90', 'runbqpb_sif.F90')
endif
galahad_examples += [['bqpbs', files('bqpbs.f90')],
                     ['bqpbs2', files('bqpbs2.f90')]]
