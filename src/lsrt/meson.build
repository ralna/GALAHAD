libgalahad_src += files('lsrt.F90')

if build_ciface
  libgalahad_c_src += files('C/lsrt_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/lsrt_pyiface.c')
endif

if libcutest.found()
  libgalahad_cutest_src += files('uselsrt.F90', 'runlsrt_sif.F90')
endif

galahad_tests += [['lsrt', 'lsrtt', files('lsrtt.F90')],
                  ['lsrt', 'lsrtti', files('lsrtti.F90')]]

galahad_c_tests += [['lsrt', 'lsrtt_c', files('C/lsrtt.c')]]

galahad_python_tests += [['lsrt', 'lsrt_py', files('Python/test_lsrt.py')]]

galahad_examples += [['lsrts', files('lsrts.f90')],
                     ['lsrts2', files('lsrts2.f90')]]
