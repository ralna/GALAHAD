if not (libmetis.found() and libmetis_version == '4')
  add_global_arguments('-Dmetis_nodend=galahad_metis', language : 'fortran')
endif

if libmetis.found() and libmetis_version == '4'
  libgalahad_src += files('metis4.f90')
elif libmetis.found() and libmetis_version == '5'
  libgalahad_src += files('metis5_adapter.c', 'metis5.f90')
else
  libgalahad_src += files('dummy_metis.f90')
endif
