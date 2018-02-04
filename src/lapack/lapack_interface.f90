! THIS VERSION: GALAHAD 2.5 - 10/06/2013 AT 15:00 GMT.

!-*-*-*-  G A L A H A D _ L A P A C K _ i n t e r f a c e   M O D U L E  -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 19th 2013

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LAPACK_interface

      IMPLICIT NONE 

      PUBLIC

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

!  LU factorization

      INTERFACE GETRF

        SUBROUTINE SGETRF( m, n, A, lda, IPIV, info )
        INTEGER :: info, lda, m, n
        INTEGER :: IPIV( * )
        REAL :: A( lda, n )
        END SUBROUTINE SGETRF

        SUBROUTINE DGETRF( m, n, A, lda, IPIV, info )
        INTEGER :: info, lda, m, n
        INTEGER :: IPIV( * )
        DOUBLE PRECISION :: A( lda, n )
        END SUBROUTINE DGETRF

      END INTERFACE GETRF

!  LU combined forward and back solves

      INTERFACE GETRS

        SUBROUTINE SGETRS( trans, n, nrhs, A, lda, IPIV, B, ldb, info )
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER :: info, lda, ldb, n, nrhs
        INTEGER :: IPIV( n )
        REAL :: A( lda, n ), B( ldb, nrhs )
        END SUBROUTINE SGETRS

        SUBROUTINE DGETRS( trans, n, nrhs, A, lda, IPIV, B, ldb, info )
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER :: info, lda, ldb, n, nrhs
        INTEGER :: IPIV( n )
        DOUBLE PRECISION :: A( lda, n ), B( ldb, nrhs )
        END SUBROUTINE DGETRS

      END INTERFACE GETRS

!  Least-squares solution

      INTERFACE GELS

        SUBROUTINE SGELS( trans, m, n, nrhs, A, lda, B, ldb, WORK, lwork, info )
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER :: m, n, nrhs, lda, ldb, lwork, info
        REAL :: A( lda, n ), B( ldb, nrhs ), WORK( lwork )
        END SUBROUTINE SGELS

        SUBROUTINE DGELS( trans, m, n, nrhs, A, lda, B, ldb, WORK, lwork, info )
        CHARACTER ( LEN = 1 ) :: trans
        INTEGER :: m, n, nrhs, lda, ldb, lwork, info 
        DOUBLE PRECISION :: A( lda, n ), B( ldb, nrhs ), WORK( lwork )
        END SUBROUTINE DGELS

      END INTERFACE GELS

!  Least-squares solution using a pivoted QR decomposition

      INTERFACE GELSY

        SUBROUTINE SGELSY( m, n, nrhs, A, lda, B, ldb, JPVT, rcond, rank,      &
                           WORK, lwork, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL :: rcond
        INTEGER :: JPVT( n )
        REAL :: A( lda, n ), B( ldb, nrhs ), WORK( * )
        END SUBROUTINE SGELSY

        SUBROUTINE DGELSY( m, n, nrhs, A, lda, B, ldb, JPVT, rcond, rank,      &
                           WORK, lwork, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        DOUBLE PRECISION :: rcond
        INTEGER :: JPVT( n )
        DOUBLE PRECISION :: A( lda, n ), B( ldb, nrhs ), WORK( * )
        END SUBROUTINE DGELSY

      END INTERFACE GELSY

!  Least-squares solution using a singular-value decomposition

      INTERFACE GELSS

        SUBROUTINE SGELSS( m, n, nrhs, A, lda, B, ldb, S, rcond, rank,         &
                           WORK, lwork, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL :: rcond
        REAL :: A( lda, n ), B( ldb, nrhs ), S( * ), WORK( * )
        END SUBROUTINE SGELSS

        SUBROUTINE DGELSS( m, n, nrhs, A, lda, B, ldb, S, rcond, rank,         &
                           WORK, lwork, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        DOUBLE PRECISION :: rcond
        DOUBLE PRECISION :: A( lda, n ), B( ldb, nrhs ), S( * ), WORK( * )
        END SUBROUTINE DGELSS

      END INTERFACE GELSS

!  Least-squares solution using a divide-and-conquor singular-value 
!  decomposition

      INTERFACE GELSD

        SUBROUTINE SGELSD( m, n, nrhs, A, lda, B, ldb, S, rcond,               &
                           rank, WORK, lwork, IWORK, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        REAL :: rcond
        INTEGER :: IWORK( * )
        REAL :: A( lda, n ), B( LDB, nrhs ), S( * ), WORK( * )
        END SUBROUTINE SGELSD

        SUBROUTINE DGELSD( m, n, nrhs, A, lda, B, ldb, S, rcond,               &
                           rank, WORK, lwork, IWORK, info )
        INTEGER :: info, lda, ldb, lwork, m, n, nrhs, rank
        DOUBLE PRECISION :: rcond
        INTEGER :: IWORK( * )
        DOUBLE PRECISION :: A( lda, n ), B( LDB, nrhs ), S( * ), WORK( * )
        END SUBROUTINE DGELSD

      END INTERFACE GELSD

!  Singular-Value Decomposition

      INTERFACE GESVD

        SUBROUTINE SGESVD( jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,     &
                           WORK, lwork, info )
        CHARACTER ( LEN = 1 ) :: jobu, jobvt
        INTEGER :: m, n, lda, ldu, ldvt, lwork, info
        REAL :: A( lda, * ), S( * ), U( ldu, * )
        REAL :: VT( ldvt, * ), WORK( lwork )
        END SUBROUTINE SGESVD

        SUBROUTINE DGESVD( jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,     &
                           WORK, lwork, info )
        CHARACTER ( LEN = 1 ) :: jobu, jobvt
        INTEGER :: m, n, lda, ldu, ldvt, lwork, info
        DOUBLE PRECISION :: A( lda, * ), S( * ), U( ldu, * )
        DOUBLE PRECISION :: VT( ldvt, * ), WORK( lwork )
        END SUBROUTINE DGESVD

      END INTERFACE GESVD

!  LDLT factorization of a tridiagonal matrix

      INTERFACE PTTRF

        SUBROUTINE SPTTRF( n, D, E, info )
        INTEGER, INTENT( IN ) :: n
        INTEGER, INTENT( OUT ) :: info
        REAL, INTENT( INOUT ) :: D( n ), E( n - 1 )
        END SUBROUTINE SPTTRF

        SUBROUTINE DPTTRF( n, D, E, info )
        INTEGER, INTENT( IN ) :: n
        INTEGER, INTENT( OUT ) :: info
        DOUBLE PRECISION, INTENT( INOUT ) :: D( n ), E( n - 1 )
        END SUBROUTINE DPTTRF

      END INTERFACE PTTRF

!  Cholesky factorization

     INTERFACE POTRF

       SUBROUTINE SPOTRF( uplo, n, A, lda, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, lda
       INTEGER, INTENT( OUT ) :: info
       REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       END SUBROUTINE SPOTRF

       SUBROUTINE DPOTRF( uplo, n, A, lda, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, lda
       INTEGER, INTENT( OUT ) :: info
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       END SUBROUTINE DPOTRF

     END INTERFACE POTRF

!  Cholesky combined forward and back solves

     INTERFACE POTRS

       SUBROUTINE SPOTRS( uplo, n, nrhs, A, lda, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE SPOTRS

       SUBROUTINE DPOTRS( uplo, n, nrhs, A, lda, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE DPOTRS

     END INTERFACE POTRS

!  Bunch-Kaufman factorization

     INTERFACE SYTRF

       SUBROUTINE SSYTRF( uplo, n, A, lda, IPIV, WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, lda, lwork
       INTEGER, INTENT( OUT ) :: info
       INTEGER, INTENT( OUT ), DIMENSION( n ) :: IPIV
       REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       REAL, INTENT( OUT ), DIMENSION( lwork ) :: WORK
       END SUBROUTINE SSYTRF

       SUBROUTINE DSYTRF( uplo, n, A, lda, IPIV, WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, lda, lwork
       INTEGER, INTENT( OUT ) :: info
       INTEGER, INTENT( OUT ), DIMENSION( n ) :: IPIV
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( OUT ), DIMENSION( lwork ) :: WORK

       END SUBROUTINE DSYTRF

     END INTERFACE SYTRF

!  Bunch-Kaufman factorization combined forward and back solves

     INTERFACE SYTRS

       SUBROUTINE SSYTRS( uplo, n, nrhs, A, lda, IPIV, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       INTEGER, INTENT( IN ), DIMENSION( n ) :: IPIV
       REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE SSYTRS

       SUBROUTINE DSYTRS( uplo, n, nrhs, A, lda, IPIV, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       INTEGER, INTENT( IN ), DIMENSION( n ) :: IPIV
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE DSYTRS

     END INTERFACE SYTRS

!  Cholesky factorization of a band matrix

     INTERFACE PBTRF

       SUBROUTINE SPBTRF( uplo, n, semi_bandwidth, A, lda, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, semi_bandwidth, lda
       INTEGER, INTENT( OUT ) :: info
       REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       END SUBROUTINE SPBTRF

       SUBROUTINE DPBTRF( uplo, n, semi_bandwidth, A, lda, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, semi_bandwidth, lda
       INTEGER, INTENT( OUT ) :: info
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       END SUBROUTINE DPBTRF

     END INTERFACE PBTRF

!  Cholesky factorization combined forward and back solves for a band matrix

     INTERFACE PBTRS

       SUBROUTINE SPBTRS( uplo, n, semi_bandwidth, nrhs, A, lda, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, semi_bandwidth, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE SPBTRS

       SUBROUTINE DPBTRS( uplo, n, semi_bandwidth, nrhs, A, lda, B, ldb, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: uplo
       INTEGER, INTENT( IN ) :: n, semi_bandwidth, nrhs, lda, ldb
       INTEGER, INTENT( OUT ) :: info
       DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B
       END SUBROUTINE DPBTRS

     END INTERFACE PBTRS

!  spectral factorization 

     INTERFACE SYEV

       SUBROUTINE SSYEV( jobz, uplo, n,  A, lda, D, WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
       INTEGER, INTENT( IN ) :: n, lda, lwork
       INTEGER, INTENT( out ) :: info
       REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       REAL, INTENT( OUT ), DIMENSION( n ) :: D
       REAL, INTENT( OUT ), DIMENSION( lwork ) :: WORK
       END SUBROUTINE SSYEV

       SUBROUTINE DSYEV( jobz, uplo, n,  A, lda, D, WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
       INTEGER, INTENT( IN ) :: n, lda, lwork
       INTEGER, INTENT( out ) :: info
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( OUT ), DIMENSION( n ) :: D
       DOUBLE PRECISION, INTENT( OUT ), DIMENSION( lwork ) :: WORK
       END SUBROUTINE DSYEV

     END INTERFACE SYEV

!  generalized spectral factorization 

     INTERFACE SYGV

       SUBROUTINE SSYGV( itype, jobz, uplo, n,  A, lda, B, ldb, D,            &
                         WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
       INTEGER, INTENT( IN ) :: itype, n, lda, ldb, lwork
       INTEGER, INTENT( out ) :: info
       REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       REAL, INTENT( INOUT ), DIMENSION( ldb, n ) :: B
       REAL, INTENT( OUT ), DIMENSION( n ) :: D
       REAL, INTENT( OUT ), DIMENSION( lwork ) :: WORK
       END SUBROUTINE SSYGV

       SUBROUTINE DSYGV( itype, jobz, uplo, n,  A, lda, B, ldb, D,            &
                         WORK, lwork, info )
       CHARACTER ( LEN = 1 ), INTENT( IN ) :: jobz, uplo
       INTEGER, INTENT( IN ) :: itype, n, lda, ldb, lwork
       INTEGER, INTENT( out ) :: info
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
       DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, n ) :: B
       DOUBLE PRECISION, INTENT( OUT ), DIMENSION( n ) :: D
       DOUBLE PRECISION, INTENT( OUT ), DIMENSION( lwork ) :: WORK
       END SUBROUTINE DSYGV

     END INTERFACE SYGV

!  eigenvalues of a Hessenberg matrix

      INTERFACE HSEQR
        SUBROUTINE SHSEQR( job, compz, n, ilo, ihi, H, ldh,  WR, WI, Z, ldz,   &
                           WORK, lwork, info )
        INTEGER, INTENT( IN ) :: ihi, ilo, ldh, ldz, lwork, n
        INTEGER, INTENT( OUT ) :: info
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: compz, job
        REAL, INTENT( INOUT ) :: H( ldh, * ), Z( ldz, * )
        REAL, INTENT( OUT ) :: WI( * ), WR( * ), WORK( * )
        END SUBROUTINE SHSEQR

        SUBROUTINE DHSEQR( job, compz, n, ilo, ihi, H, ldh,  WR, WI, Z, ldz,   &
                           WORK, lwork, info )
        INTEGER, INTENT( IN ) :: ihi, ilo, ldh, ldz, lwork, n
        INTEGER, INTENT( OUT ) :: info
        CHARACTER ( LEN = 1 ), INTENT( IN ) :: compz, job
        DOUBLE PRECISION, INTENT( INOUT ) :: H( ldh, * ), Z( ldz, * )
        DOUBLE PRECISION, INTENT( OUT ) :: WI( * ), WR( * ), WORK( * )
        END SUBROUTINE DHSEQR
      END INTERFACE HSEQR

!  End of module GALAHAD_LAPACK_interface

    END MODULE GALAHAD_LAPACK_interface
