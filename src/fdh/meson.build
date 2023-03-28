libgalahad_src += files('fdh.F90')
if libcutest.found()
  libgalahad_cutest_src += files('usefdh.F90', 'runfdh_sif.F90')
endif
galahad_examples += [['fdhs', files('fdhs.f90')],
                     ['fdhs2', files('fdhs2.f90')]]
