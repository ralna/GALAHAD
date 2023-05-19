libgalahad_src += files('lapack_interface.F90')

if not liblapack.found()
  warning('building the reference LAPACK version 3.11.0; consider providing an optimized LAPACK library')
  libgalahad_include += include_directories('v3.11')
  libgalahad_src += files('v3.11/la_constants.f90', 'v3.11/la_xisnan.F90')
  libgalahad_src += files('galahad_lapack1.f', 'galahad_lapack2.F90')
  if fc.get_id() == 'nagfor'
    libgalahad_src += files('noieeeck.f')
  else
    libgalahad_src += files('ieeeck.f')
  endif
endif
