libgalahad_src += files('dqp.F90')
galahad_binaries += [['indqp', files('indqp.F90')]]
if build_ciface
  libgalahad_c_src += files('C/dqp_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usedqp.F90', 'rundqp_sif.F90')
endif
galahad_examples += [['dqps', files('dqps.f90')],
                     ['dqps2', files('dqps2.f90')],
                     ['dqps3', files('dqps3.f90')]]
