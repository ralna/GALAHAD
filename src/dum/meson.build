dum_single_c_src = files('C/hsl_ma48s_ciface.f90', 'C/hsl_ma57s_ciface.f90', 'C/hsl_ma77s_ciface.f90',
                         'C/hsl_ma86s_ciface.f90', 'C/hsl_ma87s_ciface.f90', 'C/hsl_ma97s_ciface.f90',
                         'C/hsl_mc64s_ciface.f90', 'C/hsl_mi28s_ciface.f90', 'C/hsl_mc68i_ciface.f90')

dum_double_c_src = files('C/hsl_ma48d_ciface.f90', 'C/hsl_ma57d_ciface.f90', 'C/hsl_ma77d_ciface.f90',
                         'C/hsl_ma86d_ciface.f90', 'C/hsl_ma87d_ciface.f90', 'C/hsl_ma97d_ciface.f90',
                         'C/hsl_mc64d_ciface.f90', 'C/hsl_mi28d_ciface.f90', 'C/hsl_mc68i_ciface.f90')

if not build_ssids
  libspral_ssids_single_src += files('ssidss.F90', '../kinds/kinds.F90', '../symbols/symbols.F90')
  dum_single_c_src += files('C/ssids_ciface.F90')
  libspral_ssids_double_src += files('ssidsd.F90', '../kinds/kinds.F90', '../symbols/symbols.F90')
  dum_double_c_src += files('C/ssids_ciface.F90')
endif