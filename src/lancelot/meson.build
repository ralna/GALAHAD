libgalahad_src += files('asmbl.F90', 'cauchy.F90', 'cg.F90', 'drche.F90', 'drchg.F90',
                        'frntl.F90', 'hslint.F90', 'hsprd.F90', 'initw.F90', 'lancelot.F90',
                        'lancelot_steering.F90', 'lancelot_types.F90', 'mdchl.F90',
                        'others.F90', 'precn.F90', 'scaln.F90', 'strutr.F90')

libgalahad_cutest_src += files('uselancelot.F90', 'uselancelot_steering.F90')

galahad_cutest_binaries += [['runlancelot_sif', files('runlancelot_sif.F90')],
                            ['runlancelot_steering_sif', files('runlancelot_steering_sif.F90')]]

galahad_tests += [['lancelot', 'lancelott', files('lancelott.F90')],
                  ['lancelot', 'lancelot_steeringt', files('lancelot_steeringt.F90')]]

galahad_examples += [['lancelots', files('lancelots.f90')],
                     ['lancelot_steerings', files('lancelot_steerings.f90')]]
