libgalahad_src += files('bllsb.F90')

if build_ciface
  libgalahad_c_src += files('C/bllsb_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['bllsb', files('Python/bllsb_pyiface.c',
                                            '../fdc/Python/fdc_pyiface.c','../sls/Python/sls_pyiface.c',
                                            '../fit/Python/fit_pyiface.c','../roots/Python/roots_pyiface.c',
                                            '../cro/Python/cro_pyiface.c','../rpd/Python/rpd_pyiface.c',
                                            '../ir/Python/ir_pyiface.c')]]
endif

libgalahad_cutest_src += files('usebllsb.F90')

galahad_cutest_binaries += [['runbllsb_sif', files('runbllsb_sif.F90')]]

galahad_tests += [['bllsb', 'bllsbt', files('bllsbt.F90')],
                  ['bllsb', 'bllsbti', files('bllsbti.F90')]]

galahad_c_tests += [['bllsb', 'bllsbt_c', files('C/bllsbt.c')],
                    ['bllsb', 'bllsbtf_c', files('C/bllsbtf.c')]]

galahad_python_tests += [['bllsb', 'bllsb_py', files('Python/test_bllsb.py')]]

galahad_examples += [['bllsbs', files('bllsbs.f90')],
                     ['bllsbs2', files('bllsbs2.f90')],
                     ['bllsbs3', files('bllsbs3.f90')]]
