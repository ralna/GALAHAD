libgalahad_src += files('qp.F90')
bingalahad_src += [['inqp', files('inqp.F90')]]
libgalahad_c_src += files('C/qp_ciface.F90')
if libcutest.found()
  libgalahad_cutest_src += files('useqp.F90', 'runqp_sif.F90')
endif
