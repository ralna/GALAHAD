libgalahad_src += files('arc.F90')

if build_ciface
  libgalahad_c_src += files('C/arc_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['arc', files('Python/arc_pyiface.c')]]
endif

if libcutest.found()
  libgalahad_cutest_src += files('usearc.F90', 'runarc_sif.F90')
endif

galahad_tests += [['arc', 'arct', files('arct.F90')],
                  ['arc', 'arcti', files('arcti.F90')]]

galahad_c_tests += [['arc', 'arct_c', files('C/arct.c')],
                    ['arc', 'arctf_c', files('C/arctf.c')]]

galahad_python_tests += [['arc', 'arc_py', files('Python/test_arc.py')]]

galahad_examples += [['arcs', files('arcs.f90')],
                     ['arcs2', files('arcs2.f90')],
                     ['arcs3', files('arcs3.f90')],
                     ['arcs4', files('arcs4.f90')]]

galahad_c_examples += [['arcs_c', files('C/arcs.c')],
                       ['arcs2_c', files('C/arcs2.c')],
                       ['arcs3_c', files('C/arcs3.c')],
                       ['arcs4_c', files('C/arcs4.c')]]
