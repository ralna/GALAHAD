libgalahad_src += files('qpb.F90')

galahad_binaries += [['inqpb', files('inqpb.F90')]]

if build_ciface
  libgalahad_c_src += files('C/qpb_ciface.F90')
endif

if libcutest.found()
  libgalahad_cutest_src += files('useqpb.F90', 'runqpb_sif.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/qpb_pyiface.c')
endif

galahad_tests += [['qpb', 'qpbt', files('qpbt.F90')],
                  ['qpb', 'qpbti', files('qpbti.F90')]]

galahad_c_tests += [['qpb', 'qpbt_c', files('C/qpbt.c')],
                    ['qpb', 'qpbtf_c', files('C/qpbtf.c')]]

galahad_python_tests += [['qpb', 'qpb_py', files('Python/test_qpb.py')]]

galahad_examples += [['qpbs', files('qpbs.f90')],
                     ['qpbs2', files('qpbs2.f90')]]
