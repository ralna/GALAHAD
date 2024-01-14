libgalahad_src += files('ugo.F90')

libgalahad_c_src += files('C/ugo_ciface.F90')

libgalahad_python_src += [['ugo', files('Python/ugo_pyiface.c')]]

libgalahad_cutest_src += files('useugo.F90')

galahad_cutest_binaries += [['runugo_sif', files('runugo_sif.F90')]]

galahad_tests += [['ugo', 'ugot', files('ugot.F90')]]

galahad_c_tests += [['ugo', 'ugot_c', files('C/ugot.c')]]

galahad_python_tests += [['ugo', 'ugo_py', files('Python/test_ugo.py')]]

galahad_examples += [['ugos', files('ugos.f90')],
                     ['ugos2', files('ugos2.f90')],
                     ['ugos3', files('ugos3.f90')]]

galahad_c_examples += [['ugos_c', files('C/ugos.c')]]
