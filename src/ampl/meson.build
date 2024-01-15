if libampl.found()
  libgalahad_src += files('wrap_bqp.F90',
                          'wrap_cqp.F90',
                          'wrap_filtrane.F90',
                          'wrap_lancelot.F90',
                          'wrap_presolve.F90',
                          'wrap_qp.F90',
                          'wrap_qpa.F90',
                          'wrap_qpb.F90',
                          'wrap_qpc.F90')

  libgalahad_cc_src += files('galahad_ampl.c')

  galahad_headers += files('amplinter.h', 'galahad_ampl.h')
endif
