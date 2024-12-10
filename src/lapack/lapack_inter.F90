! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:00 GMT

#include "galahad_modules.h"
#include "galahad_lapack.h"

!-*-*-*-*-  G A L A H A D _ L A P A C K _ i n t e r    M O D U L E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 19th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LAPACK_inter_precision
      IMPLICIT NONE

      PUBLIC

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

!  parameter selection envionment

      INTERFACE LAENV

          FUNCTION ILAENV( ispec, name, opts, n1, n2, n3, n4 )
          USE GALAHAD_KINDS_precision, ONLY: ip_
          INTEGER ( KIND = ip_) :: ILAENV
          INTEGER ( KIND = ip_) :: ispec, n1, n2, n3, n4
          CHARACTER ( LEN = * ) :: name, opts
          END FUNCTION ILAENV

      END INTERFACE LAENV

!  LU factorization

      INTERFACE GETRF

        SUBROUTINE DGETRF( m, n, A, lda, IPIV, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ) :: info, lda, m, n
        INTEGER ( KIND = ip_ ) :: IPIV( * )
        REAL ( KIND = rp_ ) :: A( lda, n )
        END SUBROUTINE DGETRF

      END INTERFACE GETRF

!  LU combined forward and back solves

      INTERFACE GETRS

        SUBROUTINE DGETRS( trans, n, nrhs, A, lda, IPIV, B, ldb, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER ( KIND = ip_ ) :: info, lda, ldb, n, nrhs
        INTEGER ( KIND = ip_ ) :: IPIV( * )
        REAL ( KIND = rp_ ) :: A( lda, n ), B( ldb, nrhs )
        END SUBROUTINE DGETRS

      END INTERFACE GETRS

!  Least-squares solution

      INTERFACE GELS

        SUBROUTINE DGELS( trans, m, n, nrhs, A, lda, B, ldb, WORK, lwork, &
                          info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER ( KIND = ip_ ) :: m, n, nrhs, lda, ldb, lwork, info
        REAL ( KIND = rp_ ) :: A( lda, n ), B( ldb, nrhs ), WORK( lwork )
        END SUBROUTINE DGELS

      END INTERFACE GELS

!  Least-squares solution using a pivoted QR decomposition

      INTERFACE GELSY

        SUBROUTINE DGELSY( m, n, nrhs, A, lda, B, ldb, JPVT, rcond, rank, &
                           WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ) :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL ( KIND = rp_ ) :: rcond
        INTEGER ( KIND = ip_ ) :: JPVT( n )
        REAL ( KIND = rp_ ) :: A( lda, n ), B( ldb, nrhs ), WORK( * )
        END SUBROUTINE DGELSY

      END INTERFACE GELSY

!  Least-squares solution using a singular-value decomposition

      INTERFACE GELSS

        SUBROUTINE DGELSS( m, n, nrhs, A, lda, B, ldb, S, rcond, rank,    &
                           WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ) :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL ( KIND = rp_ ) :: rcond
        REAL ( KIND = rp_ ) :: A( lda, n ), B( ldb, nrhs ), S( * ), WORK( * )
        END SUBROUTINE DGELSS

      END INTERFACE GELSS

!  Least-squares solution using a divide-and-conquor singular-value
!  decomposition

      INTERFACE GELSD

        SUBROUTINE DGELSD( m, n, nrhs, A, lda, B, ldb, S, rcond,          &
                           rank, WORK, lwork, IWORK, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ) :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL ( KIND = rp_ ) :: rcond
        INTEGER ( KIND = ip_ ) :: IWORK( * )
        REAL ( KIND = rp_ ) :: A( lda, n ), B( LDB, nrhs ), S( * ), WORK( * )
        END SUBROUTINE DGELSD

      END INTERFACE GELSD

!  Singular-Value Decomposition

      INTERFACE GESVD

        SUBROUTINE DGESVD( jobu, jobvt, m, n, A, lda, S, U, ldu,          &
                           VT, ldvt, WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ) :: jobu, jobvt
        INTEGER ( KIND = ip_ ) :: m, n, lda, ldu, ldvt, lwork, info
        REAL ( KIND = rp_ ) :: A( lda, * ), S( * ), U( ldu, * )
        REAL ( KIND = rp_ ) :: VT( ldvt, * ), WORK( lwork )
        END SUBROUTINE DGESVD

      END INTERFACE GESVD

!  LDLT factorization of a tridiagonal matrix

      INTERFACE PTTRF

        SUBROUTINE DPTTRF( n, D, E, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ) :: D( n ), E( n - 1 )
        END SUBROUTINE DPTTRF

      END INTERFACE PTTRF

!  Cholesky factorization

      INTERFACE POTRF

        SUBROUTINE DPOTRF( uplo, n, A, lda, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lda, n ) :: A
        END SUBROUTINE DPOTRF

      END INTERFACE POTRF

!  Cholesky combined forward and back solves

      INTERFACE POTRS

        SUBROUTINE DPOTRS( uplo, n, nrhs, A, lda, B, ldb, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nrhs, lda, ldb
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
        END SUBROUTINE DPOTRS

      END INTERFACE POTRS

!  Bunch-Kaufman factorization

      INTERFACE SYTRF

        SUBROUTINE DSYTRF( uplo, n, A, lda, IPIV, WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda, lwork
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: IPIV
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( lwork ) :: WORK

        END SUBROUTINE DSYTRF

      END INTERFACE SYTRF

!  Bunch-Kaufman factorization combined forward and back solves

      INTERFACE SYTRS

        SUBROUTINE DSYTRS( uplo, n, nrhs, A, lda, IPIV, B, ldb, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nrhs, lda, ldb
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: IPIV
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
        END SUBROUTINE DSYTRS

      END INTERFACE SYTRS

!  Cholesky factorization of a band matrix

      INTERFACE PBTRF

        SUBROUTINE DPBTRF( uplo, n, semi_bandwidth, A, lda, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, lda
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lda, n ) :: A
        END SUBROUTINE DPBTRF

      END INTERFACE PBTRF

!  Cholesky factorization combined forward and back solves for a band matrix

      INTERFACE PBTRS

        SUBROUTINE DPBTRS( uplo, n, semi_bandwidth, nrhs, A, lda, B, ldb, &
                           info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, semi_bandwidth, nrhs
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: lda, ldb
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
        END SUBROUTINE DPBTRS

      END INTERFACE PBTRS

!  spectral factorization

      INTERFACE SYEV

        SUBROUTINE DSYEV( jobz, uplo, n,  A, lda, D, WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda, lwork
        INTEGER ( KIND = ip_ ), INTENT( out ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: D
        REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( lwork ) :: WORK
        END SUBROUTINE DSYEV

      END INTERFACE SYEV

!  generalized spectral factorization

      INTERFACE SYGV

        SUBROUTINE DSYGV( itype, jobz, uplo, n,  A, lda, B, ldb, D,       &
                          WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: itype, n, lda, ldb, lwork
        INTEGER ( KIND = ip_ ), INTENT( out ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lda, n ) :: A
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, n ) :: B
        REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: D
        REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( lwork ) :: WORK
        END SUBROUTINE DSYGV

      END INTERFACE SYGV

!  eigenvalues of a Hessenberg matrix

      INTERFACE HSEQR

        SUBROUTINE DHSEQR( job, compz, n, ilo, ihi, H, ldh,  WR, WI,      &
                           Z, ldz, WORK, lwork, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: ihi, ilo, ldh, ldz, lwork, n
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: compz, job
        REAL ( KIND = rp_ ), INTENT( INOUT ) :: H( ldh, * ), Z( ldz, * )
        REAL ( KIND = rp_ ), INTENT( OUT ) :: WI( * ), WR( * ), WORK( * )
        END SUBROUTINE DHSEQR

      END INTERFACE HSEQR

!  eigenvalues of a symmetric tridigonal matrix

      INTERFACE STERF

        SUBROUTINE DSTERF( n, D, E, info )
        USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info
        REAL ( KIND = rp_ ), INTENT( INOUT ) :: D( n ), E( n - 1 )
        END SUBROUTINE DSTERF

      END INTERFACE STERF

!  eigenvalues and eigenvectors of a symmetric 2 by 2 matrix

      INTERFACE LAEV2

        SUBROUTINE DLAEV2( a, b, c, rt1, rt2, cs1, sn1 )
        USE GALAHAD_KINDS_precision, ONLY: rp_
        REAL ( KIND = rp_ ), INTENT( IN ) :: a, b, c
        REAL ( KIND = rp_ ), INTENT( OUT ) :: cs1, rt1, rt2, sn1
        END SUBROUTINE DLAEV2

      END INTERFACE LAEV2

!  End of module GALAHAD_LAPACK_inter

    END MODULE GALAHAD_LAPACK_inter_precision
