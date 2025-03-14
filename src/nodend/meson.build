subdir('metis51')
subdir('metis52')

libgalahad_src += files('nodend.F90')

libgalahad_c_src += files('C/nodend_ciface.F90')

libgalahad_cc_src += files('adapter_nodend.c')

libgalahad_cutest_src += files('usenodend.F90')

galahad_cutest_binaries += [['runnodend_sif', files('runnodend_sif.F90')]]

galahad_tests += [['nodend', 'nodendt', files('nodendt.F90')]]

galahad_examples += [['nodends', files('nodends.f90')],
                     ['nodends2', files('nodends2.f90')]]
