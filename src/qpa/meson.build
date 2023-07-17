libgalahad_src += files('qpa.F90')

galahad_binaries += [['inqpa', files('inqpa.F90')]]

if build_ciface
  libgalahad_c_src += files('C/qpa_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['qpa', files('Python/qpa_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('useqpa.F90', 'runqpa_sif.F90')
endif

galahad_tests += [['qpa', 'qpat', files('qpat.F90')],
                  ['qpa', 'qpati', files('qpati.F90')]]

galahad_c_tests += [['qpa', 'qpat_c', files('C/qpat.c')],
                    ['qpa', 'qpatf_c', files('C/qpatf.c')]]

galahad_python_tests += [['qpa', 'qpa_py', files('Python/test_qpa.py')]]

galahad_examples += [['qpas', files('qpas.f90')]]
