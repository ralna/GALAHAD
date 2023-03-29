libgalahad_src += files('qpa.F90')
galahad_binaries += [['inqpa', files('inqpa.F90')]]
if build_ciface
  libgalahad_c_src += files('C/qpa_ciface.F90')
endif
if libcutest.found()
  libgalahad_cutest_src += files('useqpa.F90', 'runqpa_sif.F90')
endif
galahad_examples += [['qpas', files('qpas.f90')]]
