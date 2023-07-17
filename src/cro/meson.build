libgalahad_src += files('cro.F90')

if build_ciface
  libgalahad_c_src += files('C/cro_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['cro', files('Python/cro_pyiface.c')]]
endif

galahad_tests += [['cro', 'crot', files('crot.F90')],
                  ['cro', 'croti', files('croti.F90')]]

galahad_c_tests += [['cro', 'crot_c', files('C/crot.c')],
                    ['cro', 'crotf_c', files('C/crotf.c')]]

galahad_python_tests += [['cro', 'cro_py', files('Python/test_cro.py')]]

galahad_examples += [['cros', files('cros.f90')],
                     ['cros2', files('cros2.f90')],
                     ['cros3', files('cros3.f90')]]
