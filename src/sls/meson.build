libgalahad_src += files('sls.F90')

if build_ciface
  libgalahad_c_src += files('C/sls_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['sls', files('Python/sls_pyiface.c',
                                          '../sils/Python/sils_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('usesls.F90', 'runsls_sif.F90')
endif

galahad_tests += [['sls', 'slst', files('slst.F90')],
                  ['sls', 'slsti', files('slsti.F90')]]

galahad_c_tests += [['sls', 'slst_c', files('C/slst.c')],
                    ['sls', 'slstf_c', files('C/slstf.c')]]

galahad_python_tests += [['sls', 'sls_py', files('Python/test_sls.py')]]

galahad_examples += [['slss', files('slss.f90')]]
