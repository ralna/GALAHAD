libgalahad_src += files('bsc.F90')

libgalahad_c_src += files('C/bsc_ciface.F90')

libgalahad_python_src += [['bsc', files('Python/bsc_pyiface.c')]]

galahad_tests += [['bsc', 'bsct', files('bsct.F90')],
                  ['bsc', 'bscti', files('bscti.F90')]]

galahad_c_tests += [['bsc', 'bsct_c', files('C/bsct.c')],
                    ['bsc', 'bsctf_c', files('C/bsctf.c')]]

galahad_python_tests += [['bsc', 'bsc_py', files('Python/test_bsc.py')]]

galahad_examples += [['bscs', files('bscs.f90')]]
