libgalahad_src += files('llsr.F90')

if build_ciface
  libgalahad_c_src += files('C/llsr_ciface.F90')
endif

galahad_tests += [['llsr', 'llsrt', files('llsrt.F90')],
                  ['llsr', 'llsrti', files('llsrti.F90')]]

galahad_c_tests += [['llsr', 'llsrt_c', files('C/llsrt.c')],
                    ['llsr', 'llsrtf_c', files('C/llsrtf.c')]]

galahad_examples += [['llsrs', files('llsrs.f90')],
                     ['llsrs2', files('llsrs2.f90')],
                     ['llsrs3', files('llsrs3.f90')]]
