libgalahad_src += files('blas_interface.F90')

if not libblas.found()
  warning('building the reference BLAS version 3.11.0; consider providing an optimized BLAS library')
  libgalahad_src += files('cgemv.f', 'cher2k.f', 'cscal.f', 'ctbmv.f', 'ctrsv.f',
                          'dgemm.f', 'drotmg.f', 'dswap.f', 'dtbmv.f', 'dtrsv.f',
                          'lsame.f', 'scopy.f', 'snrm2.f90', 'sspmv.f', 'ssyr2k.f',
                          'strmm.f', 'zcopy.f', 'zgemv.f', 'zher2k.f', 'zscal.f', 'ztpmv.f',
                          'caxpy.f', 'cgerc.f', 'cher.f', 'csrot.f', 'ctbsv.f', 'dasum.f',
                          'dgemv.f', 'dsbmv.f', 'dsymm.f', 'dtbsv.f', 'dzasum.f', 'sdot.f',
                          'srot.f', 'sspr2.f', 'ssyr.f', 'strmv.f', 'zdotc.f', 'zgerc.f', 'zher.f',
                          'zswap.f', 'ztpsv.f', 'ccopy.f', 'cgeru.f', 'cherk.f', 'csscal.f', 'ctpmv.f',
                          'daxpy.f', 'dger.f', 'dscal.f', 'dsymv.f', 'dtpmv.f', 'dznrm2.f90', 'sasum.f',
                          'sdsdot.f', 'srotg.f90', 'sspr.f', 'ssyrk.f', 'strsm.f', 'zdotu.f', 'zgeru.f',
                          'zherk.f', 'zsymm.f', 'ztrmm.f', 'cdotc.f', 'chbmv.f', 'chpmv.f', 'cswap.f',
                          'ctpsv.f', 'dcabs1.f', 'dnrm2.f90', 'dsdot.f', 'dsyr2.f', 'dtpsv.f', 'icamax.f',
                          'saxpy.f', 'sgbmv.f', 'srotm.f', 'sswap.f', 'stbmv.f', 'strsv.f', 'zdrot.f',
                          'zhbmv.f', 'zhpmv.f', 'zsyr2k.f', 'ztrmv.f', 'cdotu.f', 'chemm.f', 'chpr2.f',
                          'csymm.f', 'ctrmm.f', 'dcopy.f', 'drot.f', 'dspmv.f', 'dsyr2k.f', 'dtrmm.f',
                          'idamax.f', 'scabs1.f', 'sgemm.f', 'srotmg.f', 'ssymm.f', 'stbsv.f',
                          'xerbla_array.f', 'zdscal.f', 'zhemm.f', 'zhpr2.f', 'zsyrk.f', 'ztrsm.f',
                          'cgbmv.f', 'chemv.f', 'chpr.f', 'csyr2k.f', 'ctrmv.f', 'ddot.f', 'drotg.f90',
                          'dspr2.f', 'dsyr.f', 'dtrmv.f', 'isamax.f', 'scasum.f', 'sgemv.f', 'ssbmv.f',
                          'ssymv.f', 'stpmv.f', 'xerbla.f', 'zgbmv.f', 'zhemv.f', 'zhpr.f', 'ztbmv.f',
                          'ztrsv.f', 'cgemm.f', 'cher2.f', 'crotg.f90', 'csyrk.f', 'ctrsm.f', 'dgbmv.f',
                          'drotm.f', 'dspr.f', 'dsyrk.f', 'dtrsm.f', 'izamax.f', 'scnrm2.f90', 'sger.f',
                          'sscal.f', 'ssyr2.f', 'stpsv.f', 'zaxpy.f', 'zgemm.f', 'zher2.f', 'zrotg.f90',
                          'ztbsv.f')
endif
