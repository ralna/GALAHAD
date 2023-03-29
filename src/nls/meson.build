libgalahad_src += files('nls.F90')
if build_ciface
  libgalahad_c_src += files('C/nls_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usenls.F90', 'runnls_sif.F90')
endif

galahad_tests += [['nls', 'nlst', files('nlst.F90')],
                  ['nls', 'nlsti', files('nlsti.F90')]]

galahad_c_tests += [['nls', 'nlst_c', files('C/nlst.c')],
                    ['nls', 'nlstf_c', files('C/nlstf.c')]]

galahad_examples += [['nlss', files('nlss.f90')],
                     ['nlss2', files('nlss2.f90')],
                     ['nlss3', files('nlss3.f90')]]

galahad_c_examples += [['nlss_c', files('C/nlss.c')],
                       ['nlss2_c', files('C/nlss2.c')],
                       ['nlss3_c', files('C/nlss3.c')],
                       ['nlss4_c', files('C/nlss4.c')]]
