libgalahad_c_single_src += files('C/hsl_ma48s_ciface.f90', 'C/hsl_ma57s_ciface.f90', 'C/hsl_ma77s_ciface.f90',
                                 'C/hsl_ma86s_ciface.f90', 'C/hsl_ma87s_ciface.f90', 'C/hsl_ma97s_ciface.f90',
                                 'C/hsl_mc64s_ciface.f90', 'C/hsl_mi28s_ciface.f90')

libgalahad_c_double_src += files('C/hsl_ma48d_ciface.f90', 'C/hsl_ma57d_ciface.f90', 'C/hsl_ma77d_ciface.f90',
                                 'C/hsl_ma86d_ciface.f90', 'C/hsl_ma87d_ciface.f90', 'C/hsl_ma97d_ciface.f90',
                                 'C/hsl_mc64d_ciface.f90', 'C/hsl_mi28d_ciface.f90')

libgalahad_c_integer_src += files('C/hsl_mc68i_ciface.f90')

if not libcutest.found()
  libgalahad_cutest_src += files('cutest_dummy.F90')
endif

if not libhsl.found()
  libgalahad_single_src += files('fa14s.f', 'fd15s.f', 'hsl_ad02s.f90', 'hsl_ma48s.f90', 'hsl_ma54s.f90', 'hsl_ma57s.f90',
                                 'hsl_ma64s.f90', 'hsl_ma77s.f90', 'hsl_ma86s.f90', 'hsl_ma87s.f90', 'hsl_ma97s.f90',
                                 'hsl_mc34s.f90', 'hsl_mc64s.f90', 'hsl_mc65s.f90', 'ma48s.f', 'mc64s.f', 'mc71s.f',
                                 'hsl_mc68s.f90', 'ma50s.f', 'mc77s.f', 'ma51s.f', 'hsl_mc69s.f90', 'ma57s.f', 'mi21s.f',
                                 'hsl_mc80s.f90', 'ma61s.f', 'hsl_mi20s.f90', 'mi24s.f', 'hsl_mi28s.f90', 'mc13s.f', 'mi26s.f',
                                 'hsl_mi32s.f90', 'mc20s.f', 'hsl_mi35s.f90', 'mc21s.f', 'hsl_of01s.f90', 'mc22s.f', 'mc23s.f',
                                 'hsl_zb01s.f90', 'mc29s.f', 'mc30s.f', 'la04s.f', 'la15s.f', 'mc34s.f', 'ma27s.f', 'mc47s.f',
                                 'mc59s.f', 'zb01s.f', 'mc60s.f', 'mc61s.f', 'ma33s.f')  # 'ma33a.f'

  libgalahad_double_src += files('fa14d.f', 'fd15d.f', 'hsl_ad02d.f90', 'hsl_ma48d.f90', 'hsl_ma54d.f90', 'hsl_ma57d.f90',
                                 'hsl_ma64d.f90', 'hsl_ma77d.f90', 'hsl_ma86d.f90', 'hsl_ma87d.f90', 'hsl_ma97d.f90',
                                 'hsl_mc34d.f90', 'hsl_mc64d.f90', 'hsl_mc65d.f90', 'ma48d.f', 'mc64d.f', 'mc71d.f',
                                 'hsl_mc68d.f90', 'ma50d.f', 'mc77d.f', 'ma51d.f', 'hsl_mc69d.f90', 'ma57d.f', 'mi21d.f',
                                 'hsl_mc80d.f90', 'ma61d.f', 'hsl_mi20d.f90', 'mi24d.f', 'hsl_mi28d.f90', 'mc13d.f', 'mi26d.f',
                                 'hsl_mi32d.f90', 'mc20d.f', 'hsl_mi35d.f90', 'mc21d.f', 'hsl_of01d.f90', 'mc22d.f', 'mc23d.f',
                                 'hsl_zb01d.f90', 'mc29d.f', 'mc30d.f', 'la04d.f', 'la15d.f', 'mc34d.f', 'ma27d.f', 'mc47d.f',
                                 'mc59d.f', 'zb01d.f', 'mc60d.f', 'mc61d.f', 'ma33d.f')  # 'ma33ad.f'

  libgalahad_integer_src += files('hsl_kb22l.f90', 'hsl_mc68i.f90', 'hsl_mc78i.f90', 'hsl_of01i.f90', 'hsl_zb01i.f90', 'kb07i.f')
endif

if not libmetis.found()
  libgalahad_src += files('metis4.f')
endif

if not libwsmp.found()
  libgalahad_src += files('wsmp.F90')
endif

if not libumfpack.found()
  libgalahad_src += files('umfpack.F90')
endif

if not libpardiso.found()
  libgalahad_src += files('pardiso.F90')
endif

if not libspmf.found()
  libgalahad_src += files('spmf.F90', 'spmf_enums.F90')
endif

if not libpastix.found()
  libgalahad_src += files('pastixf.F90', 'pastixf_enums.F90')
endif

if not libsmumps.found()
  libgalahad_single_src += files('mumps.F90')
endif

if not libdmumps.found()
  libgalahad_double_src += files('mumps.F90')
endif

if not libmkl_pardiso.found()
  libgalahad_src += files('mkl_pardiso.F90')
endif

if not libmpi.found()
  libgalahad_src += files('mpi.F90')
endif

if not build_ssids
  libgalahad_single_src += files('ssidss.F90')
  libgalahad_double_src += files('ssidsd.F90')
  libgalahad_c_src += files('C/ssids_ciface.F90')
endif
