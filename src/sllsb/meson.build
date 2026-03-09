libgalahad_src += files('sllsb.F90')

libgalahad_c_src += files('C/sllsb_ciface.F90')

libgalahad_python_src += [['sllsb', files('Python/sllsb_pyiface.c',
                                         '../nodend/Python/nodend_pyiface.c',
                                         '../fdc/Python/fdc_pyiface.c',
                                         '../sls/Python/sls_pyiface.c',
                                         '../fit/Python/fit_pyiface.c',
                                         '../roots/Python/roots_pyiface.c',
                                         '../cro/Python/cro_pyiface.c',
                                         '../rpd/Python/rpd_pyiface.c',
                                         '../ir/Python/ir_pyiface.c',
                                         '../uls/Python/uls_pyiface.c',
                                         '../sils/Python/sils_pyiface.c',
                                         '../sbls/Python/sbls_pyiface.c',
                                         '../gls/Python/gls_pyiface.c')]]

libgalahad_cutest_src += files('usesllsb.F90')

galahad_cutest_binaries += [['runsllsb_sif', files('runsllsb_sif.F90')]]

galahad_tests += [['sllsb', 'sllsbt', files('sllsbt.F90')],
                  ['sllsb', 'sllsbti', files('sllsbti.F90')]]

galahad_c_tests += [['sllsb', 'sllsbt_c', files('C/sllsbt.c')],
                    ['sllsb', 'sllsbtf_c', files('C/sllsbtf.c')]]

galahad_python_tests += [['sllsb', 'sllsb_py', files('Python/test_sllsb.py')]]

galahad_examples += [['sllsbs', files('sllsbs.f90')],
                     ['sllsbs2', files('sllsbs2.f90')],
                     ['sllsbs3', files('sllsbs3.f90')],
                     ['sllsbs4', files('sllsbs4.f90')]]
