libgalahad_src += files('lsrt.F90')

libgalahad_c_src += files('C/lsrt_ciface.F90')

libgalahad_python_src += [['lsrt', files('Python/lsrt_pyiface.c')]]

libgalahad_cutest_src += files('uselsrt.F90')

galahad_cutest_binaries += [['runlsrt_sif', files('runlsrt_sif.F90')]]

galahad_tests += [['lsrt', 'lsrtt', files('lsrtt.F90')],
                  ['lsrt', 'lsrtti', files('lsrtti.F90')]]

galahad_c_tests += [['lsrt', 'lsrtt_c', files('C/lsrtt.c')]]

galahad_python_tests += [['lsrt', 'lsrt_py', files('Python/test_lsrt.py')]]

galahad_examples += [['lsrts', files('lsrts.f90')],
                     ['lsrts2', files('lsrts2.f90')]]
