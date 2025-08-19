! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:20 GMT

! Define BLAS API in Fortran module

#include "ssids_routines.h"

MODULE GALAHAD_SSIDS_blas_iface
  IMPLICIT none

  PRIVATE
  PUBLIC :: sgemv, strsv
  PUBLIC :: dgemv, dtrsv
  PUBLIC :: sgemm, ssyrk, strsm
  PUBLIC :: dgemm, dsyrk, dtrsm

!  Level 2 BLAS

  INTERFACE
    SUBROUTINE sgemv( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
      USE GALAHAD_KINDS, ONLY: ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: trans
      INTEGER( ip_) , INTENT( IN ) :: m, n, lda, incx, incy
      REAL( sp_ ), INTENT( IN ) :: alpha, beta
      REAL( sp_) , INTENT( IN ), DIMENSION( lda, n ) :: a
      REAL( sp_) , INTENT( IN ), DIMENSION( * ) :: x
      REAL( sp_ ), INTENT( INOUT ), DIMENSION( * ) :: y
    END SUBROUTINE sgemv
    SUBROUTINE strsv( uplo, trans, diag, n, a, lda, x, incx )
      USE GALAHAD_KINDS, ONLY: ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo, trans, diag
      INTEGER( ip_ ), INTENT( IN ) :: n, lda, incx
      REAL( sp_ ), INTENT( IN ), DIMENSION( lda, n ) :: a
      REAL( sp_ ), INTENT( INOUT ), DIMENSION( * ) :: x
    END SUBROUTINE strsv
  END INTERFACE

  INTERFACE
    SUBROUTINE dgemv( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
      USE GALAHAD_KINDS, ONLY: ip_, dp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: trans
      INTEGER( ip_ ), INTENT( IN ) :: m, n, lda, incx, incy
      REAL( dp_ ), INTENT( IN ) :: alpha, beta
      REAL( dp_ ), INTENT( IN ), DIMENSION( lda, n ) :: a
      REAL( dp_ ), INTENT( IN ), DIMENSION( * ) :: x
      REAL( dp_ ), INTENT( INOUT ), DIMENSION( * ) :: y
    END SUBROUTINE dgemv
    SUBROUTINE dtrsv( uplo, trans, diag, n, a, lda, x, incx )
      USE GALAHAD_KINDS, ONLY: ip_, dp_
      implicit none
      CHARACTER, INTENT( IN ) :: uplo, trans, diag
      INTEGER( ip_ ), INTENT( IN ) :: n, lda, incx
      REAL( dp_ ), INTENT( IN ), DIMENSION( lda, n ) :: a
      REAL( dp_ ), INTENT( INOUT ), DIMENSION( * ) :: x
    END SUBROUTINE dtrsv
  END INTERFACE

! Level 3 BLAS

  INTERFACE
    SUBROUTINE sgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      USE GALAHAD_KINDS, ONLY : ip_, sp_
      IMPLICIT NONE
      CHARACTER, INTENT( IN ) :: ta, tb
      INTEGER( ip_ ), INTENT( IN ) :: m, n, k
      INTEGER( ip_ ), INTENT( IN ) :: lda, ldb, ldc
      REAL( sp_ ), INTENT( IN ) :: alpha, beta
      REAL( sp_ ), INTENT( IN ), DIMENSION( lda, * ) :: a
      REAL( sp_ ), INTENT( IN ), DIMENSION( ldb, * ) :: b
      REAL( sp_ ), INTENT( INOUT ), DIMENSION( ldc, * ) :: c
    END SUBROUTINE sgemm
    SUBROUTINE ssyrk( uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      USE GALAHAD_KINDS, ONLY : ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo, trans
      INTEGER( ip_ ), INTENT( IN ) :: n, k, lda, ldc
      REAL( sp_ ), INTENT( IN ) :: alpha, beta
      REAL( sp_ ), INTENT( IN ), DIMENSION( lda, * ) :: a
      REAL( sp_ ), INTENT( INOUT ), DIMENSION( ldc, n ) :: c
    END SUBROUTINE ssyrk
    SUBROUTINE strsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      USE GALAHAD_KINDS, ONLY : ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: side, uplo, trans, diag
      INTEGER( ip_ ), INTENT( IN ) :: m, n, lda, ldb
      REAL( sp_ ), INTENT( IN ) :: alpha
      REAL( sp_ ), INTENT( IN ) :: a(l da, * )
      REAL( sp_ ), INTENT( INOUT ) :: b( ldb, n )
    END SUBROUTINE strsm
  END INTERFACE

  INTERFACE
    SUBROUTINE dgemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
      USE GALAHAD_KINDS, ONLY : ip_, dp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: ta, tb
      INTEGER( ip_ ), INTENT( IN ) :: m, n, k
      INTEGER( ip_ ), INTENT( IN ) :: lda, ldb, ldc
      REAL( dp_ ), INTENT( IN ) :: alpha, beta
      REAL( dp_ ), INTENT( IN ), DIMENSION( lda, * ) :: a
      REAL( dp_ ), INTENT( IN ), DIMENSION( ldb, * ) :: b
      REAL( dp_ ), INTENT( INOUT ), DIMENSION( ldc, * ) :: c
    END SUBROUTINE dgemm
    SUBROUTINE dsyrk( uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      USE GALAHAD_KINDS, ONLY : ip_, dp_
      IMPLICIT NONE
      CHARACTER, INTENT( IN ) :: uplo, trans
      INTEGER( ip_ ), INTENT( IN ) :: n, k, lda, ldc
      REAL( dp_ ), INTENT( IN ) :: alpha, beta
      REAL( dp_ ), INTENT( IN ), DIMENSION( lda, * ) :: a
      REAL( dp_ ), INTENT( INOUT ), DIMENSION( ldc, n ) :: c
    END SUBROUTINE dsyrk
    SUBROUTINE dtrsm( side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb )
      USE GALAHAD_KINDS, ONLY : ip_, dp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: side, uplo, trans, diag
      INTEGER( ip_ ), INTENT( IN ) :: m, n, lda, ldb
      REAL( dp_ ), INTENT( IN ) :: alpha
      REAL( dp_ ), INTENT( IN ) :: a( lda, * )
      REAL( dp_ ), INTENT( INOUT ) :: b( ldb, n )
    END SUBROUTINE dtrsm
  END INTERFACE

END MODULE GALAHAD_SSIDS_blas_iface
