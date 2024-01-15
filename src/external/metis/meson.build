if libmetis.found() and libmetis_version == '4'
  libgalahad_src += files('metis4.f90')
elif libmetis.found() and libmetis_version == '5'
  libgalahad_src += files('metis5.f90')
  libgalahad_cc_src += files('metis5_adapter.c')
else
  libgalahad_src += files('dummy_metis.f90')
endif
