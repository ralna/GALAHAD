libgalahad_src += files('cqp.F90')
bingalahad_src += [['incqp', files('incqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/cqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usecqp.F90', 'runcqp_sif.F90')
endif
galahad_examples += [['cqps', files('cqps.f90')],
                     ['cqps2', files('cqps2.f90')],
                     ['cqps3', files('cqps3.f90')]]
