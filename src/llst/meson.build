libgalahad_src += files('llst.F90')

if build_ciface
  libgalahad_c_src += files('C/llst_ciface.F90')
endif

galahad_tests += [['llst', 'llstt', files('llstt.F90')],
                  ['llst', 'llstti', files('llstti.F90')]]

galahad_c_tests += [['llst', 'llstt_c', files('C/llstt.c')],
                    ['llst', 'llsttf_c', files('C/llsttf.c')]]

galahad_examples += [['llsts', files('llsts.f90')],
                     ['llsts2', files('llsts2.f90')],
                     ['llsts3', files('llsts3.f90')]]
