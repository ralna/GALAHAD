libgalahad_src += files('dgo.F90')

if build_ciface
  libgalahad_c_src += files('C/dgo_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/dgo_pyiface.c')
endif

if libcutest.found()
  libgalahad_cutest_src += files('usedgo.F90', 'rundgo_sif.F90')
endif

galahad_tests += [['dgo', 'dgot', files('dgot.F90')],
                  ['dgo', 'dgoti', files('dgoti.F90')]]

galahad_c_tests += [['dgo', 'dgot_c', files('C/dgot.c')],
                    ['dgo', 'dgotf_c', files('C/dgotf.c')]]

galahad_python_tests += [['dgo', 'dgo_py', files('Python/test_dgo.py')],
                         ['dgo', 'dgo_quadratic_py', files('Python/test_dgo_quadratic.py')]]

galahad_examples += [['dgos', files('dgos.f90')],
                     ['dgos2', files('dgos2.f90')]]

galahad_c_examples += [['dgos_c', files('C/dgos.c')],
                       ['dgos2_c', files('C/dgos2.c')],
                       ['dgos3_c', files('C/dgos3.c')],
                       ['dgos4_c', files('C/dgos4.c')]]
