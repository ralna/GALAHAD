libgalahad_src += files('fit.F90')

if build_ciface
  libgalahad_c_src += files('C/fit_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['fit', files('Python/fit_pyiface.c')]]
endif

galahad_tests += [['fit', 'fitt', files('fitt.F90')]]

galahad_python_tests += [['fit', 'fit_py', files('Python/test_fit.py')]]

galahad_examples += [['fits', files('fits.f90')]]
