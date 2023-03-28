libgalahad_src += files('arc.F90')
if build_ciface
  libgalahad_c_src += files('C/arc_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('usearc.F90', 'runarc_sif.F90')
endif
galahad_c_examples += [['arcs_c', files('C/arcs.c')],
                       ['arcs2_c', files('C/arcs2.c')],
                       ['arcs3_c', files('C/arcs3.c')],
                       ['arcs4_c', files('C/arcs4.c')]]
