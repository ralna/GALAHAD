if build_ssids
  libgalahad_cpp_src += files('cholesky.cxx', 'ldlt_app.cxx', 'ldlt_nopiv.cxx',
                              'ldlt_tpp.cxx', 'NumericSubtree.cxx', 
                              'profile.cxx',  'SymbolicSubtree.cxx', 
                              'ThreadStats.cxx', 'wrappers.cxx')

  libgalahad_src += files('akeep.F90', 'anal.F90', 
                          'blas_iface.F90', 'ssids.F90',
                          'contrib.F90', 'contrib_free.F90', 'cpu_iface.F90',
                          'cpu_solve.F90', 'cpu_subtree.F90', 'datatypes.F90',
                          'fkeep.F90', 'gpu_subtree_no_cuda.F90', 'inform.F90',
                          'lapack_iface.F90', 'profile_iface.F90', 
                          'subtree.F90')

  libgalahad_c_src += files('C/ssids_ciface.F90')

  galahad_tests += [['ssids', 'ssidst', files('ssidst.F90')]]

  galahad_examples += [['ssidss', files('ssidss.f90')]]
endif

# GPU: 'assemble.cu', 'dense_factor.cu', 'reorder.cu', 'solve.cu', 'syrk.cu'
# 'gpu_alloc.F90', 'gpu_datatypes.F90',
# 'gpu_dense_factor.F90', 'gpu_factor.F90', 'gpu_interfaces.F90',
# 'gpu_smalloc.F90', 'gpu_solve.F90', 'gpu_subtree.F90'
