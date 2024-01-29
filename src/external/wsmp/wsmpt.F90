! THIS VERSION: GALAHAD 4.3 - 2024-01-31 AT 07:50 GMT.

#include "galahad_modules.h"

PROGRAM TEST_WSMP

  USE GALAHAD_KINDS_precision
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE

!  variables

  INTEGER ( KIND = ip_ ), PARAMETER :: n = 8
  INTEGER ( KIND = ip_ ), PARAMETER :: ldb = n
  INTEGER ( KIND = ip_ ), PARAMETER :: nrhs = 1
  INTEGER ( KIND = ip_ ), PARAMETER :: naux = 0
  INTEGER ( KIND = ip_ ), PARAMETER :: num_threads = 4
  INTEGER ( KIND = ip_ ) :: nz, error
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: IA, JA, PERM, INVP
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION ( : ) :: IPARM, MRP
  REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION ( : ) :: A, DIAG, AUX, DPARM
  REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION ( : , : ) :: B, X
  INTEGER ( KIND = ip_ ) :: i, j, l, idum( 1 )
  REAL( KIND = rp_ ) :: ddum( 1 )
  REAL( KIND = rp_ ) :: bdum( 1, 1 )
  LOGICAL :: all_in_one = .FALSE.
  LOGICAL :: easy_problem = .FALSE.

  ALLOCATE( AUX( naux ), DIAG( 0 ), IPARM( 64 ), DPARM( 64 ) )
  ALLOCATE( IA( n + 1 ), PERM( n ), INVP( n ), MRP( n ) )
  ALLOCATE( B( ldb, nrhs ), X( ldb, nrhs ) )

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
           -1.0_rp_, 5.0_rp_, 11.0_rp_, 5.0_rp_ /)
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

  CALL WSETMAXTHRDS( num_threads )
  CALL WSMP_INITIALIZE( )
  IPARM( 1 ) = 0
  IPARM( 2 ) = 0
  IPARM( 3 ) = 0
  CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, B, ldb, nrhs, aux, naux, MRP,    &
              IPARM, DPARM )

!  check for error returns

  error = IPARM( 64 )
  IF ( error == GALAHAD_unavailable_option ) THEN
    WRITE( 6, "( ' WSMP is not available' )" )
    GO TO 1
  ELSE IF ( error /= 0 ) THEN
    WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
    GO TO 1
  ELSE
    WRITE( 6, "( ' initialization completed ... ' )" )
  END IF

!  Either solve the problem in a single phase ...

  IF ( all_in_one ) THEN
    IPARM( 2 ) = 1
    IPARM( 3 ) = 5
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, B, ldb, nrhs, aux, naux, MRP,  &
                IPARM, DPARM )

!  check for error returns

    error = IPARM( 64 )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    ELSE
      WRITE( 6, "( ' all-in-one factorize and solve completed ... ' )" )
    END IF

!  ... or in a sequence of phases

  ELSE

!  reordering and Symbolic Factorization, This step also allocates all memory
!  that is necessary for the factorization

    IPARM( 2 ) = 1
    IPARM( 3 ) = 2
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, bdum, 1_ip_, 1_ip_, aux, naux, &
                MRP, IPARM, DPARM )

!  check for error returns

    error = IPARM( 64 )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected (analyse): ', I0 )" )    &
        error
      GO TO 1
    ELSE
      WRITE( 6, "( ' reordering and symbolic facorization completed ... ' )" )
    END IF
    WRITE( 6, "( ' number of nonzeros in factors = 1000 * ', I0 )" ) IPARM( 24 )
    WRITE( 6, "( ' number of factorization Mflops = ', F5.2 )" ) DPARM( 23 )

!  factorization

    IPARM( 2 ) = 3
    IPARM( 3 ) = 3
    IPARM( 31 ) = 1
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, bdum, 1_ip_, 1_ip_, aux, naux, &
                MRP, IPARM, DPARM )

!  check for error returns

    error = IPARM( 64 )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected (factorize): ', I0 )" )  &
        error
      GO TO 1
    ELSE
      WRITE( 6, "( ' factorization completed ... ' )" )
    END IF

!  back substitution and iterative refinement

    IPARM( 2 ) = 4
    IPARM( 3 ) = 5
    X = B
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, X, ldb, nrhs, aux, naux, MRP,  &
                IPARM, DPARM )

!  check for error returns

    error = IPARM( 64 )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    ELSE
      WRITE( 6, "( ' solve completed ... ' )" )
    END IF

  END IF

  WRITE( 6, "( ' the solution of the system is X =' )" )
  WRITE( 6, "( ( 4ES12.4 ) )" ) X( : n, 1 )

!  termination and release of memory

1 CONTINUE
  DEALLOCATE( IA, JA, A, DIAG, X, B, PERM, INVP, AUX, MRP, IPARM, DPARM )
  CALL WSMP_CLEAR( )
  CALL WSSFREE( )

END PROGRAM TEST_WSMP
