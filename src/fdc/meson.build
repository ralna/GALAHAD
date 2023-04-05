libgalahad_src += files('fdc.F90')

if build_ciface
  libgalahad_c_src += files('C/fdc_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/fdc_pyiface.c')
endif

galahad_tests += [['fdc', 'fdct', files('fdct.F90')],
                  ['fdc', 'fdcti', files('fdcti.F90')]]

galahad_c_tests += [['fdc', 'fdct_c', files('C/fdct.c')],
                    ['fdc', 'fdctf_c', files('C/fdctf.c')]]

galahad_python_tests += [['fdc', 'fdc_py', files('Python/test_fdc.py')]]

galahad_examples += [['fdcs', files('fdcs.f90')]]
