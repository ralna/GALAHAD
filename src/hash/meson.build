libgalahad_src += files('hash.F90')

if build_ciface
  libgalahad_c_src += files('C/hash_ciface.F90')
endif

if build_pythoniface
  libgalahad_python_src += [['hash', files('Python/hash_pyiface.c')]]
endif

galahad_tests += [['hash', 'hasht', files('hasht.F90')]]

galahad_python_tests += [['hash', 'hash_py', files('Python/test_hash.py')]]

galahad_examples += [['hashs', files('hashs.f90')]]
