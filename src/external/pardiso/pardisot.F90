#include "galahad_modules.h"

PROGRAM TEST_PARDISO

  USE GALAHAD_KINDS_precision
  USE GALAHAD_SYMBOLS

  IMPLICIT NONE

!  variables

  INTEGER ( KIND = ip_ ), PARAMETER :: n = 8
  INTEGER ( KIND = ip_ ), PARAMETER :: nrhs = 1
  INTEGER ( KIND = ip_ ) :: nz, maxfct, mnum, mtype, phase, error, msglvl
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: IA, JA, IPARM
  INTEGER ( KIND = long_ ), ALLOCATABLE, DIMENSION( : ) :: PT
  REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION ( : ) :: A, DPARM
  REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION ( : , : ) :: B, X
  INTEGER ( KIND = ip_ ) :: i, j, l, idum( 1 )
  REAL( KIND = rp_ ) :: ddum( 1 )
  LOGICAL :: all_in_one = .FALSE.
  LOGICAL :: easy_problem = .FALSE.

  ALLOCATE( IPARM( 64 ), DPARM( 64 ), PT( 64 ) )
  ALLOCATE( IA( n + 1 ), B( n, nrhs ), X( n, nrhs ) )

!  fill all arrays containing matrix data

  IF ( easy_problem ) THEN
    nz = n
    ALLOCATE( A( nz ), JA( nz ) )
    DO i = 1, n
      IA( i ) = i ; JA( i ) = i ; A( i ) = 1.0_rp_
    END DO
    IA( n + 1 ) = n + 1
  ELSE
    nz = 18
    ALLOCATE( A( nz ), JA( nz ) )
    IA = (/ 1, 5, 8, 10, 12, 15, 17, 18, 19 /)
    JA = (/ 1, 3, 6, 7, 2, 3, 5, 3, 8, 4, 7, 5, 6, 7, 6, 8, 7, 8 /)
    A = (/ 7.0_rp_, 1.0_rp_, 2.0_rp_, 7.0_rp_, -4.0_rp_, 8.0_rp_, 2.0_rp_,     &
           1.0_rp_, 5.0_rp_, 7.0_rp_, 9.0_rp_, 5.0_rp_, 1.0_rp_, 5.0_rp_,      &
           - 1.0_rp_, 5.0_rp_, 11.0_rp_, 5.0_rp_ /)
  END IF

!  set the RHS so that the required solution is a vector of ones

! B = 1.0_rp_ ! set right-hand side
  B = 0.0_rp_
  DO i = 1, n
    DO l = IA( i ), IA( i + 1 ) - 1
      j = JA( l )
      B( i, 1 ) = B( i, 1 ) + A( l )
      IF ( i /= j ) B( j, 1 ) = B( j, 1 ) + A( l )
    END DO
  END DO

!  set defaults

  mtype = - 2 ! symmetric, indefinite
! mtype = 1 ! symmetric
! mtype = 2 ! symmetric, positive definite
  CALL PARDISOINIT( PT, mtype, 0, IPARM, DPARM, error )

!  check for error returns

  IF ( error == GALAHAD_unavailable_option ) THEN
    WRITE( 6, "( ' PARDISO is not available' )" )
    GO TO 1
  ELSE IF ( error /= 0 ) THEN
    WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
    GO TO 1
  ELSE
    WRITE( 6, "( ' initialization completed ... ' )" )
  END IF

  maxfct = 1
  mnum = 1

  error = 0 ! initialize error flag
  msglvl = 0 ! don't print statistical information
! msglvl = 1 ! print statistical information

!  Either solve the problem in a single phase ...

  IF ( all_in_one ) THEN
    phase = 13 ! complete solution
    CALL PARDISO( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,      &
                            idum, nrhs, IPARM, msglvl, B, X, error, DPARM )
    IF ( error == GALAHAD_unavailable_option ) THEN
      WRITE( 6, "( ' PARDISO is not available' )" )
      GO TO 1
    ELSE IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    ELSE
      WRITE( 6, "( ' all-in-one factorize and solve completed ... ' )" )
    END IF

!  ... or in a sequence of phases

  ELSE

!  reordering and Symbolic Factorization, This step also allocates all memory
!  that is necessary for the factorization

    phase = 11 ! only reordering and symbolic factorization
    CALL PARDISO( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,                &
                  idum, nrhs, IPARM, msglvl, ddum, ddum, error, DPARM )

    IF ( error == GALAHAD_unavailable_option ) THEN
      WRITE( 6, "( ' MKL PARDISO is not available' )" )
      GO TO 1
    ELSE IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    ELSE
      WRITE( 6, "( ' reordering and symbolic facorization completed ... ' )" )
    END IF
    WRITE( 6, "( ' number of nonzeros in factors = ', I0 )" ) IPARM( 18 )
    WRITE( 6, "( ' number of factorization Mflops = ', I0 )" ) IPARM( 19 )

!  factorization

    phase = 22 ! only factorization
    CALL PARDISO( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,                &
                  idum, nrhs, IPARM, msglvl, ddum, ddum, error, DPARM )

    WRITE( 6, "( ' factorization completed ... ' )" )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    END IF

!  back substitution and iterative refinement

    iparm(8) = 2 ! max numbers of iterative refinement steps
    phase = 33 ! only solution
    CALL PARDISO( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,                &
                  idum, nrhs, IPARM, msglvl, B, X, error, DPARM )

    WRITE( 6, * ) ' solve completed ... '
  END IF

  WRITE( 6, "( ' the solution of the system is X =' )" )
  WRITE( 6, "( ( 4ES12.4 ) )" ) X( : n, 1 )

!  termination and release of memory

1 CONTINUE
  phase = - 1 ! release internal memory
  CALL PARDISO( PT, maxfct, mnum, mtype, phase, n, ddum, idum, idum,           &
                idum, nrhs, IPARM, msglvl, ddum, ddum, error, DPARM )
  DEALLOCATE( IPARM, DPARM, IA, JA, A, B, X, PT )

END PROGRAM TEST_PARDISO
