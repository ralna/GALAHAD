libgalahad_src += files('trb.F90')

if build_ciface
  libgalahad_c_src += files('C/trb_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += files('Python/trb_pyiface.c')
endif

if libcutest.found()
  libgalahad_cutest_src += files('usetrb.F90', 'runtrb_sif.F90')
endif

galahad_tests += [['trb', 'trbt', files('trbt.F90')],
                  ['trb', 'trbti', files('trbti.F90')]]

galahad_c_tests += [['trb', 'trbt_c', files('C/trbt.c')],
                    ['trb', 'trbtf_c', files('C/trbtf.c')]]

galahad_examples += [['trbs', files('trbs.f90')],
                     ['trbs2', files('trbs2.f90')],
                     ['trbs3', files('trbs3.f90')]]

galahad_c_examples += [['trbs_c', files('C/trbs.c')],
                       ['trbs2_c', files('C/trbs2.c')],
                       ['trbs3_c', files('C/trbs3.c')],
                       ['trbs4_c', files('C/trbs4.c')]]
