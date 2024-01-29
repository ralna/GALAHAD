! THIS VERSION: GALAHAD 4.3 - 2024-01-27 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_blas.h"

!-*-*-*-  G A L A H A D _ L A P A C K _ i n t e r f a c e   M O D U L E  -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 19th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BLAS_interface

     IMPLICIT NONE

     PUBLIC

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

!  two norm

     INTERFACE NRM2

       FUNCTION SNRM2( n, X, incx )
       USE GALAHAD_KINDS_precision
       REAL :: SNRM2
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
       END FUNCTION SNRM2

       FUNCTION DNRM2( n, X, incx )
       USE GALAHAD_KINDS_precision
       DOUBLE PRECISION :: DNRM2
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
       END FUNCTION DNRM2

     END INTERFACE NRM2

!  compute plane rotation

     INTERFACE ROTG

       SUBROUTINE SROTG( a, b, c, s )
       USE GALAHAD_KINDS_precision
       REAL, INTENT( INOUT ) :: a, b
       REAL, INTENT( OUT ) :: c, s
       END SUBROUTINE SROTG

       SUBROUTINE DROTG( a, b, c, s )
       USE GALAHAD_KINDS_precision
       DOUBLE PRECISION, INTENT( INOUT ) :: a, b
       DOUBLE PRECISION, INTENT( OUT ) :: c, s
       END SUBROUTINE DROTG

     END INTERFACE ROTG

!  apply plane rotation

     INTERFACE ROT

       SUBROUTINE SROT( n, X, incx, Y, incy, c, s )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       REAL, INTENT( IN ) :: c, s
       REAL, INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE SROT

       SUBROUTINE DROT( n, X, incx, Y, incy, c, s )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       DOUBLE PRECISION, INTENT( IN ) :: c, s
       DOUBLE PRECISION, INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE DROT

     END INTERFACE ROT

!  swap vectors

     INTERFACE SWAP

       SUBROUTINE SSWAP( n, X, incx, Y, incy )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       REAL, INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE SSWAP

       SUBROUTINE DSWAP( n, X, incx, Y, incy )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx, incy
       DOUBLE PRECISION, INTENT( INOUT ) :: X( * ), Y( * )
       END SUBROUTINE DSWAP

     END INTERFACE SWAP

!  scale a vector by a constant

     INTERFACE SCAL

       SUBROUTINE SSCAL( n, sa, SX, incx )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL, INTENT( IN ) :: sa
       REAL, INTENT( INOUT ) :: SX( * )
       END SUBROUTINE SSCAL

       SUBROUTINE DSCAL( n, sa, SX, incx )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       DOUBLE PRECISION, INTENT( IN ) :: sa
       DOUBLE PRECISION, INTENT( INOUT ) :: SX( * )
       END SUBROUTINE DSCAL

     END INTERFACE SCAL

!  index of element having maximum absolute value

     INTERFACE IAMAX

       FUNCTION ISAMAX( n, X, incx )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ) :: ISAMAX
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
       END FUNCTION ISAMAX

       FUNCTION IDAMAX( n, X, incx )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ) :: IDAMAX
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, incx
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
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
       USE GALAHAD_KINDS
       CHARACTER ( LEN = * ) :: srname
       INTEGER ( KIND = ip_ ) :: info
       END SUBROUTINE XERBLA

     END INTERFACE XERBLA

!  triangular solve

     INTERFACE TRSV

       SUBROUTINE STRSV( uplo, trans, diag, n, A, lda, X, incx )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda, incx
       REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ( n - 1 ) * incx + 1 ) :: X
       END SUBROUTINE STRSV

       SUBROUTINE DTRSV( uplo, trans, diag, n, A, lda, X, incx )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda, incx
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ( n - 1 ) * incx + 1 ) :: X
       END SUBROUTINE DTRSV

     END INTERFACE TRSV

!  block triangular solve

     INTERFACE TRSM

       SUBROUTINE STRSM( side, uplo, transa, diag, m, n, alpha, A, lda,    &    
                         B, ldb )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: side, uplo, transa, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, lda, ldb
       REAL, INTENT( IN ) :: alpha
       REAL, INTENT( IN ), DIMENSION( lda, * ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ldb, * ) :: B
       END SUBROUTINE STRSM

       SUBROUTINE DTRSM( side, uplo, transa, diag, m, n, alpha, A, lda,    &
                         B, ldb )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: side, uplo, transa, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, lda, ldb
       DOUBLE PRECISION, INTENT( IN ) :: alpha
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, * ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, * ) :: B
       END SUBROUTINE DTRSM

     END INTERFACE TRSM

!  triangular solve for a band matrix

     INTERFACE TBSV

       SUBROUTINE STBSV( uplo, trans, diag, n, semi_bandwidth, A, lda,     &
                         X, incx )
       USE GALAHAD_KINDS_precision
!       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
!       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda, incx
!       REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
!       REAL, INTENT( INOUT ), DIMENSION( ( n - 1 ) * incx + 1 ) :: X
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda, incx
       REAL, INTENT( IN ) :: A( lda, * )
       REAL, INTENT( INOUT ) :: X( * )
       END SUBROUTINE STBSV

       SUBROUTINE DTBSV( uplo, trans, diag, n, semi_bandwidth, A, lda,    &
                         X, incx )
       USE GALAHAD_KINDS_precision
!      CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
!      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda, incx
!      DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
!      DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ( n - 1 ) * incx + 1 ) :: X
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo, trans, diag
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda, incx
       DOUBLE PRECISION, INTENT( IN ) :: A( lda, * )
       DOUBLE PRECISION, INTENT( INOUT ) :: X( * )
       END SUBROUTINE DTBSV

     END INTERFACE TBSV

!  matrix-vector product

     INTERFACE GEMV

       SUBROUTINE SGEMV( trans, m, n, alpha, A, lda, X, incx, beta,       &
                         Y, incy )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: trans
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       REAL, INTENT( IN ) :: alpha, beta
       REAL, INTENT( IN ) :: A( lda, * ), X( * )
       REAL, INTENT( INOUT ) :: Y( * )
       END SUBROUTINE SGEMV

       SUBROUTINE DGEMV( trans, m, n, alpha, A, lda, X, incx, beta,       &
                         Y, incy )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: trans
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       DOUBLE PRECISION, INTENT( IN ) :: alpha, beta
       DOUBLE PRECISION, INTENT( IN ) :: A( lda, * ), X( * )
       DOUBLE PRECISION, INTENT( INOUT ) :: Y( * )
       END SUBROUTINE DGEMV

     END INTERFACE GEMV

!  matrix-matrix product

     INTERFACE GEMM

       SUBROUTINE SGEMM( transa, transb, m, n, k, alpha, A, lda, B, ldb,   &
                         beta, C, ldc )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: transa, transb
       INTEGER ( KIND = ip_ ), INTENT( IN )  :: m, n, k, lda, ldb, ldc
       REAL, INTENT( IN ) :: alpha, beta
       REAL, INTENT( IN ) :: A( lda, * ), B( ldb, * )
       REAL, INTENT( INOUT ) :: C( ldc, * )
       END SUBROUTINE SGEMM

       SUBROUTINE DGEMM( transa, transb, m, n, k, alpha, A, lda, B, ldb,   &
                         beta, C, ldc )
       USE GALAHAD_KINDS_precision
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: transa, transb
       INTEGER ( KIND = ip_ ), INTENT( IN )  :: m, n, k, lda, ldb, ldc
       DOUBLE PRECISION, INTENT( IN ) :: alpha, beta
       DOUBLE PRECISION, INTENT( IN ) :: A( lda, * ), B( ldb, * )
       DOUBLE PRECISION, INTENT( INOUT ) :: C( ldc, * )
       END SUBROUTINE DGEMM

     END INTERFACE GEMM

!  rank-one update

     INTERFACE GER

       SUBROUTINE SGER( m, n, alpha, X, incx, Y, incy, A, lda )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       REAL, INTENT( IN ) :: alpha
       REAL, INTENT( IN ) :: X( * ), Y( * )
       REAL, INTENT( INOUT ) :: A( lda, * )
       END SUBROUTINE SGER

       SUBROUTINE DGER( m, n, alpha, X, incx, Y, incy, A, lda )
       USE GALAHAD_KINDS_precision
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: incx, incy, lda, m, n
       DOUBLE PRECISION, INTENT( IN ) :: alpha
       DOUBLE PRECISION, INTENT( IN ) :: X( * ), Y( * )
       DOUBLE PRECISION, INTENT( INOUT ) :: A( lda, * )
       END SUBROUTINE DGER

     END INTERFACE GER

!  End of module GALAHAD_BLAS_interface

   END MODULE GALAHAD_BLAS_interface
