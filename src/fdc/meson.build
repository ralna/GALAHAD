libgalahad_src += files('fdc.F90')
if build_ciface
  libgalahad_c_src += files('C/fdc_ciface.F90')
endif

galahad_tests += [['fdc', 'fdct', files('fdct.F90')],
                  ['fdc', 'fdcti', files('fdcti.F90')]]

galahad_c_tests += [['fdc', 'fdct_c', files('C/fdct.c')],
                    ['fdc', 'fdctf_c', files('C/fdctf.c')]]

galahad_examples += [['fdcs', files('fdcs.f90')]]
