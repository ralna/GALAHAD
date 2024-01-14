libgalahad_src += files('arc.F90')

libgalahad_c_src += files('C/arc_ciface.F90')

libgalahad_python_src += [['arc', files('Python/arc_pyiface.c',
                                        '../sls/Python/sls_pyiface.c','../ir/Python/ir_pyiface.c','../rqs/Python/rqs_pyiface.c',
                                        '../glrt/Python/glrt_pyiface.c','../psls/Python/psls_pyiface.c','../dps/Python/dps_pyiface.c',
                                        '../lms/Python/lms_pyiface.c','../sha/Python/sha_pyiface.c','../sils/Python/sils_pyiface.c',
                                        '../rpd/Python/rpd_pyiface.c')]]

libgalahad_cutest_src += files('usearc.F90')

galahad_cutest_binaries += [['runarc_sif', files('runarc_sif.F90')]]

galahad_tests += [['arc', 'arct', files('arct.F90')],
                  ['arc', 'arcti', files('arcti.F90')]]

galahad_c_tests += [['arc', 'arct_c', files('C/arct.c')],
                    ['arc', 'arctf_c', files('C/arctf.c')]]

galahad_python_tests += [['arc', 'arc_py', files('Python/test_arc.py')]]

galahad_examples += [['arcs', files('arcs.f90')],
                     ['arcs2', files('arcs2.f90')],
                     ['arcs3', files('arcs3.f90')],
                     ['arcs4', files('arcs4.f90')]]

galahad_c_examples += [['arcs_c', files('C/arcs.c')],
                       ['arcs2_c', files('C/arcs2.c')],
                       ['arcs3_c', files('C/arcs3.c')],
                       ['arcs4_c', files('C/arcs4.c')]]
