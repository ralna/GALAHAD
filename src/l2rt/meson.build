libgalahad_src += files('l2rt.F90')

libgalahad_c_src += files('C/l2rt_ciface.F90')

libgalahad_python_src += [['l2rt', files('Python/l2rt_pyiface.c')]]

libgalahad_cutest_src += files('usel2rt.F90')

galahad_cutest_binaries += [['runl2rt_sif', files('runl2rt_sif.F90')]]

galahad_tests += [['l2rt', 'l2rtt', files('l2rtt.F90')],
                  ['l2rt', 'l2rtti', files('l2rtti.F90')]]

galahad_c_tests += [['l2rt', 'l2rtt_c', files('C/l2rtt.c')]]

galahad_python_tests += [['l2rt', 'l2rt_py', files('Python/test_l2rt.py')]]

galahad_examples += [['l2rts', files('l2rts.f90')],
                     ['l2rts2', files('l2rts2.f90')]]
