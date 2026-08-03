libgalahad_src += files('ssids.F90', 'contrib_iface.F90',
                        'types.F90', 'subtree.F90',
                        'ssids_factor.F90', 'numeric_subtree.F90')

libgalahad_c_src += files('C/ssids_ciface.F90')

galahad_tests += [['ssids', 'ssidst', files('ssidst.F90')],
                  ['ssids', 'ssids_factort', files('ssids_factort.F90')]]

galahad_c_tests += [['ssids', 'ssidst_c', files('C/ssidst.c')],
                    ['ssids', 'ssidstf_c', files('C/ssidstf.c')]]

galahad_examples += [['ssidss', files('ssidss.f90')]]
