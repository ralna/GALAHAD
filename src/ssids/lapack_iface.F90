! THIS VERSION: GALAHAD 5.3 - 2025-08-14 AT 10:20 GMT

! Definition of LAPACK API in module

#ifdef INTEGER_64
#define GALAHAD_SSIDS_lapack_iface GALAHAD_SSIDS_lapack_iface_64
#define GALAHAD_KINDS GALAHAD_KINDS_64
#endif

MODULE GALAHAD_SSIDS_lapack_iface
  IMPLICIT none

  PRIVATE
  PUBLIC :: spotrf, ssytrf
  PUBLIC :: dpotrf, dsytrf

  INTERFACE
    SUBROUTINE spotrf( uplo, n, a, lda, info )
      USE GALAHAD_KINDS, ONLY: ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo
      INTEGER( ip_ ), INTENT( IN ) :: n, lda
      REAL( sp_ ), INTENT( INOUT ) :: a(lda, n)
      INTEGER( ip_ ), INTENT( INOUT ) :: info
    END SUBROUTINE spotrf
    SUBROUTINE ssytrf( uplo, n, a, lda, ipiv, work, lwork, info )
      USE GALAHAD_KINDS, ONLY: ip_, sp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo
      INTEGER( ip_ ), INTENT( IN ) :: n, lda, lwork
      INTEGER( ip_ ), INTENT( INOUT ), dimension(n) :: ipiv
      INTEGER( ip_ ), INTENT( INOUT ) :: info
      REAL( sp_ ), INTENT( INOUT ), dimension(lda, *) :: a
      REAL( sp_ ), intent(out  ), dimension(*) :: work
    END SUBROUTINE ssytrf
  END INTERFACE

  INTERFACE
    SUBROUTINE dpotrf( uplo, n, a, lda, info )
      USE GALAHAD_KINDS, ONLY: ip_, dp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo
      INTEGER( ip_ ), INTENT( IN ) :: n, lda
      REAL( dp_ ), INTENT( INOUT ) :: a(lda, n)
      INTEGER( ip_ ), INTENT( INOUT ) :: info
    END SUBROUTINE dpotrf
    SUBROUTINE dsytrf( uplo, n, a, lda, ipiv, work, lwork, info )
      USE GALAHAD_KINDS, ONLY: ip_, dp_
      IMPLICIT none
      CHARACTER, INTENT( IN ) :: uplo
      INTEGER( ip_ ), INTENT( IN ) :: n, lda, lwork
      INTEGER( ip_ ), INTENT( INOUT ), dimension(n) :: ipiv
      INTEGER( ip_ ), INTENT( INOUT ) :: info
      REAL( dp_ ), INTENT( INOUT ), dimension(lda, *) :: a
      REAL( dp_ ), intent(out  ), dimension(*) :: work
    END SUBROUTINE dsytrf
  END INTERFACE

END MODULE GALAHAD_SSIDS_lapack_iface
