! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:00 GMT

#include "galahad_modules.h"
#include "galahad_blas.h"

!-*-*-*-*-  G A L A H A D _ L A P A C K _ i n t e r    M O D U L E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 19th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BLAS_inter_precision

     IMPLICIT NONE

     PUBLIC

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

!  two norm

     INTERFACE NRM2

       FUNCTION DNRM2( n, X, incx )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       REAL ( KIND = rp_ ) :: DNRM2
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
       END FUNCTION DNRM2

     END INTERFACE NRM2

!  compute plane rotation

     INTERFACE ROTG

       SUBROUTINE DROTG( a, b, c, s )
       USE GALAHAD_KINDS_precision, ONLY: rp_
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: a, b
       REAL ( KIND = rp_ ), INTENT( OUT ) :: c, s
       END SUBROUTINE DROTG

     END INTERFACE ROTG

!  apply plane rotation

     INTERFACE ROT

       SUBROUTINE DROT( n, X, incx, Y, incy, c, s )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       REAL ( KIND = rp_ ), INTENT( IN ) :: c, s
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE DROT

     END INTERFACE ROT

!  swap vectors

     INTERFACE SWAP

       SUBROUTINE DSWAP( n, X, incx, Y, incy )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE DSWAP

     END INTERFACE SWAP

!  scale a vector by a constant

     INTERFACE SCAL

       SUBROUTINE DSCAL( n, sa, SX, incx )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL ( KIND = rp_ ), INTENT( IN ) :: sa
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: SX( * )
       END SUBROUTINE DSCAL

     END INTERFACE SCAL

!  index of element having maximum absolute value

     INTERFACE IAMAX

       FUNCTION IDAMAX( n, X, incx )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ) :: IDAMAX
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
       END FUNCTION IDAMAX

     END INTERFACE IAMAX

!  case comparison

     INTERFACE LSAME

       FUNCTION LSAME( ca, cb )
       USE GALAHAD_KINDS
       LOGICAL :: LSAME
       CHARACTER :: ca, cb
       END FUNCTION LSAME

     END INTERFACE LSAME

!  error handler

     INTERFACE XERBLA

       SUBROUTINE XERBLA( srname, info )
       USE GALAHAD_KINDS, ONLY: ip_
       CHARACTER ( LEN = * ) :: srname
       INTEGER ( KIND = ip_ ) :: info
       END SUBROUTINE XERBLA

     END INTERFACE XERBLA

!  triangular solve

     INTERFACE TRSV

       SUBROUTINE DTRSV( uplo, trans, diag, n, A, lda, X, incx )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda, incx
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
       REAL ( KIND = rp_ ), INTENT( INOUT ),                              &
         DIMENSION( ( n - 1 ) * incx + 1 ) :: X
       END SUBROUTINE DTRSV

     END INTERFACE TRSV

!  block triangular solve

     INTERFACE TRSM

       SUBROUTINE DTRSM( side, uplo, transa, diag, m, n, alpha, A, lda,    &
                         B, ldb )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: side, uplo, transa, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, lda, ldb
       REAL ( KIND = rp_ ), INTENT( IN ) :: alpha
       REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, * ) :: A
       REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, * ) :: B
       END SUBROUTINE DTRSM

     END INTERFACE TRSM

!  triangular solve for a band matrix

     INTERFACE TBSV

       SUBROUTINE DTBSV( uplo, trans, diag, n, semi_bandwidth, A, lda,    &
                         X, incx )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda, incx
       REAL ( KIND = rp_ ), INTENT( IN ) :: A( lda, * )
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: X( * )
!      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
!      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ( n - 1 ) * incx + 1)::X
       END SUBROUTINE DTBSV

     END INTERFACE TBSV

!  matrix-vector product

     INTERFACE GEMV

       SUBROUTINE DGEMV( trans, m, n, alpha, A, lda, X, incx, beta,       &
                         Y, incy )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: trans
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       REAL ( KIND = rp_ ), INTENT( IN ) :: alpha, beta
       REAL ( KIND = rp_ ), INTENT( IN ) :: A( lda, * ), X( * )
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: Y( * )
       END SUBROUTINE DGEMV

     END INTERFACE GEMV

!  matrix-matrix product

     INTERFACE GEMM

       SUBROUTINE DGEMM( transa, transb, m, n, k, alpha, A, lda, B, ldb,   &
                         beta, C, ldc )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: transa, transb
       INTEGER ( KIND = ip_ ), INTENT( IN )  :: m, n, k, lda, ldb, ldc
       REAL ( KIND = rp_ ), INTENT( IN ) :: alpha, beta
       REAL ( KIND = rp_ ), INTENT( IN ) :: A( lda, * ), B( ldb, * )
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: C( ldc, * )
       END SUBROUTINE DGEMM

     END INTERFACE GEMM

!  rank-one update

     INTERFACE GER

       SUBROUTINE DGER( m, n, alpha, X, incx, Y, incy, A, lda )
       USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       REAL ( KIND = rp_ ), INTENT( IN ) :: alpha
       REAL ( KIND = rp_ ), INTENT( IN ) :: X( * ), Y( * )
       REAL ( KIND = rp_ ), INTENT( INOUT ) :: A( lda, * )
       END SUBROUTINE DGER

     END INTERFACE GER

!  symmetric rank-k update

  INTERFACE SYRK

    SUBROUTINE DSYRK( uplo, trans, n, k, alpha, A, lda, beta, C, ldc )
    USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
    CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, k, lda, ldc
    REAL ( KIND = rp_ ), INTENT( IN ) :: alpha, beta
    REAL ( KIND = rp_ ), INTENT( IN ) :: A( lda, * )
    REAL ( KIND = rp_ ), INTENT( INOUT ) :: C( ldc, * )
    END SUBROUTINE DSYRK

  END INTERFACE SYRK

!  End of module GALAHAD_BLAS_inter

   END MODULE GALAHAD_BLAS_inter_precision
