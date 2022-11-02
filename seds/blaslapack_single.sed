s/D+0/E+0/g
s/(C_DOUBLE)/(C_FLOAT)/g
s/ double / float /g
s/ double\* / float* /g
s/<double>/<float>/g
s/daxpy/saxpy/g
s/dcopy/scopy/g
s/ddot/sdot/g
s/dnrm2/snrm2/g
s/dscal/sscal/g
s/zaxpy/caxpy/g
s/zcopy/ccopy/g
s/zdotc/cdotc/g
s/dznrm2/scnrm2/g
s/zscal/cscal/g
s/dgemv/sgemv/g
s/dtrsv/strsv/g
s/dgemm/sgemm/g
s/dsyrk/ssyrk/g
s/dtrsm/strsm/g
s/zgemm/cgemm/g
s/ztrsm/ctrsm/g
s/dpotrf/spotrf/g
s/dlacpy/slacpy/g
s/dsytrf/ssytrf/g
s/zpotrf/cpotrf/g
s/zlacpy/clacpy/g
