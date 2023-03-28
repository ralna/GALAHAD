libgalahad_src += files('bqp.F90')
bingalahad_src += [['inbqp', files('inbqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/bqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usebqp.F90', 'runbqp_sif.F90')
endif
galahad_examples += [['bqps', files('bqps.f90')],
                     ['bqps2', files('bqps2.f90')],
                     ['bqps3', files('bqps3.f90')]]
