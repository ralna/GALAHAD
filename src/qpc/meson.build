libgalahad_src += files('qpc.F90')
bingalahad_src += [['inqpc', files('inqpc.F90')]]
if build_ciface
  libgalahad_c_src += files('C/qpc_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useqpc.F90', 'runqpc_sif.F90')
endif
galahad_examples += [['qpcs', files('qpcs.f90')]]
