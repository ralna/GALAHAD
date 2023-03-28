libgalahad_src += files('qp.F90')
bingalahad_src += [['inqp', files('inqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/qp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useqp.F90', 'runqp_sif.F90')
endif
