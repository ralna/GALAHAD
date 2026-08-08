libgalahad_src += files('slblt.F90', 'contrib_iface.F90',
                        'types.F90', 'subtree.F90',
                        'slblt_factor.F90', 'numeric_subtree.F90')

libgalahad_c_src += files('C/slblt_ciface.F90')

galahad_tests += [['slblt', 'slbltt', files('slbltt.F90')],
                  ['slblt', 'slblt_factort', files('slblt_factort.F90')]]

galahad_c_tests += [['slblt', 'slbltt_c', files('C/slbltt.c')],
                    ['slblt', 'slblttf_c', files('C/slblttf.c')]]

galahad_examples += [['slblts', files('slblts.f90')]]
