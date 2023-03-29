libgalahad_src += files('qpb.F90')
galahad_binaries += [['inqpb', files('inqpb.F90')]]
if build_ciface
  libgalahad_c_src += files('C/qpb_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useqpb.F90', 'runqpb_sif.F90')
endif
galahad_examples += [['qpbs', files('qpbs.f90')],
                     ['qpbs2', files('qpbs2.f90')]]
