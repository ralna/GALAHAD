libgalahad_src += files('blas_interface.F90')

if not libblas.found()
  warning('building the reference BLAS version 3.11.0; consider providing an optimized BLAS library')
  libgalahad_include += include_directories('v3.11')
  libgalahad_src += files('galahad_blas1.f', 'galahad_blas2.F90')
endif
