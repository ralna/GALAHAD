libgalahad_src += files('eqp.F90')

if build_ciface
  libgalahad_c_src += files('C/eqp_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['eqp', files('Python/eqp_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('useeqp.F90', 'runeqp_sif.F90')
endif

galahad_tests += [['eqp', 'eqpt', files('eqpt.F90')],
                  ['eqp', 'eqpti', files('eqpti.F90')]]

galahad_c_tests += [['eqp', 'eqpt_c', files('C/eqpt.c')],
                    ['eqp', 'eqptf_c', files('C/eqptf.c')]]

galahad_python_tests += [['eqp', 'eqp_py', files('Python/test_eqp.py')]]

galahad_examples += [['eqps', files('eqps.f90')]]
