libgalahad_src += files('mkl_pardiso/mkl_pardiso_interface.F90', 'mumps/mumps_types.F90',
                        'pastix/spmf_interfaces.F90', 'pastix/pastixf_interfaces.F90')

galahad_tests += [['mkl_pardiso', 'mkl_pardisot', files('mkl_pardiso/mkl_pardisot.F90')],
                  ['mumps', 'mumpst', files('mumps/mumpst.F90')],
                  ['pardiso', 'pardisot', files('pardiso/pardisot.F90')],
                  ['wsmp', 'wsmpt', files('wsmp/wsmpt.F90')],
                  ['pastix', 'pastixt', files('pastix/pastixt.F90')]]

# galahad_examples += [['ma48ss', files('ma48/ma48ss.f90')],
#                      ['ma48ds', files('ma48/ma48ds.f90')]]
