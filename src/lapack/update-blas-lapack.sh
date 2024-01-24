#!/bin/sh
BLAS=./lapack/BLAS/SRC
LAPACK=./lapack/SRC
INSTALL=./lapack/INSTALL

# BLAS
mkdir tmpBLAS
for func in ddot sdot dcopy scopy dscal sscal daxpy saxpy dnrm2 snrm2 \
            drot srot drotg srotg dgemm sgemm idamax isamax \
            dtpsv stpsv dswap sswap dgemv sgemv dtrsm strsm dtrsv strsv \
            dsyrk ssyrk dasum sasum dtrmv strmv dger sger dsyr ssyr \
            dtrmm strmm dsymv ssymv dsyr2k ssyr2k dsyr2 ssyr2 dsymm ssymm \
            dtbsv stbsv dtpmv stpmv zscal zgemm zher zswap ztrsm zdotc \
            zgemv zdscal dcabs1 zgeru dznrm2 zcopy ztrmm ztrmv zgerc \
            lsame xerbla; do
    cp $BLAS/$func.f tmpBLAS/
done
cat tmpBLAS/*.f > blas_original.f
rm -rf tmpBLAS

# LAPACK
mkdir tmpLAPACK
# Double and single precision (prefixed by d or s)
for unc in bdsqr gebd2 gebrd gelq2 gelqf gelsd gels gelss gelsy geqp3 \
           geqr2 geqrf gesvd getf2 getrf getrf2 getrs isnan labad labrd \
           lacpy lae2 laed6 laev2 laic1 laisnan lals0 lalsa lalsd lamrg \
           lange lanst lansy lapy2 laqp2 laqps larfb larf larfg larft \
           lartg larzb larz larzt las2 lascl lasd4 lasd5 lasd6 lasd7 \
           lasd8 lasda lasdq lasdt laset lasq1 lasq2 lasq3 lasq4 lasq5 \
           lasq6 lasr lasrt lassq lasv2 laswp lasyf latrd latrz org2l \
           org2r orgbr orgl2 orglq orgql orgqr orgtr orm2r ormbr orml2 \
           ormlq ormqr ormr3 ormrz pbtf2 pbtrf pbtrs potf2 potrf potrf2 \
           potrs  pttrf pttrs ptts2 rscl  steqr sterf syev sygs2 sygst \
           sygv sytd2 sytf2 sytrd sytrf sytrs trtrs tzrzf hseqr gehd2 \
           gehrd laexc lahqr lahr2 lanv2 laqr0 laqr1 laqr2 laqr3 laqr4 \
           laqr5 larfx lasy2 ormhr trexc combssq; do
    cp $LAPACK/s$unc.f tmpLAPACK/
    cp $LAPACK/d$unc.f tmpLAPACK/
done
# Generic or complex
for func in iladlc iladlr ilaenv ilaslc ilaslr iparmq dladiv dlapy3 \
            ilazlc ilazlr zlacgv zladiv zlarfb zlarf zlarfg zlarft; do
    cp $LAPACK/$func.f tmpLAPACK/
done
# In Install folder
for func in slamch dlamch; do
    cp $INSTALL/$func.f tmpLAPACK/
done
cat tmpLAPACK/*.f > lapack_original.f
rm -rf tmpLAPACK

# Copy to GALAHAD
cp blas_original.f lapack_original.f $LAPACK/ieeeck.f $GALAHAD/src/lapack/
