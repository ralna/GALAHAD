libgalahad_src += files('cqp.F90')

galahad_binaries += [['incqp', files('incqp.F90')]]

if build_ciface
  libgalahad_c_src += files('C/cqp_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['cqp', files('Python/cqp_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('usecqp.F90', 'runcqp_sif.F90')
endif

galahad_tests += [['cqp', 'cqpt', files('cqpt.F90')],
                  ['cqp', 'cqpti', files('cqpti.F90')]]

galahad_c_tests += [['cqp', 'cqpt_c', files('C/cqpt.c')],
                    ['cqp', 'cqptf_c', files('C/cqptf.c')]]

galahad_python_tests += [['cqp', 'cqp_py', files('Python/test_cqp.py')]]

galahad_examples += [['cqps', files('cqps.f90')],
                     ['cqps2', files('cqps2.f90')],
                     ['cqps3', files('cqps3.f90')]]
