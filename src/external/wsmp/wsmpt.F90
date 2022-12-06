PROGRAM TEST_WSMP

  USE GALAHAD_SYMBOLS
  IMPLICIT NONE

!  precision

  INTEGER, PARAMETER :: wp = KIND( 1.0D0 )
  INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!  variables

  INTEGER, PARAMETER :: n = 8
  INTEGER, PARAMETER :: ldb = n
  INTEGER, PARAMETER :: nrhs = 1
  INTEGER, PARAMETER :: naux = 0
  INTEGER, PARAMETER :: num_threads = 4
  INTEGER:: nz, error
  INTEGER, ALLOCATABLE, DIMENSION ( : ) :: IA, JA, PERM, INVP, IPARM, MRP
  REAL( KIND = wp ), ALLOCATABLE, DIMENSION ( : ) :: A, DIAG, AUX, DPARM
  REAL( KIND = wp ), ALLOCATABLE, DIMENSION ( : , : ) :: B, X
  INTEGER :: i, j, l, idum( 1 )
  REAL( KIND = wp ) :: ddum( 1 )
  REAL( KIND = wp ) :: bdum( 1, 1 )
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
      IA( i ) = i ; JA( i ) = i ; A( i ) = 1.0_wp
    END DO
    IA( n + 1 ) = n + 1
  ELSE 
    nz = 18
    ALLOCATE( A( nz ), JA( nz ) )
    IA = (/ 1, 5, 8, 10, 12, 15, 17, 18, 19 /)
    JA = (/ 1, 3, 6, 7, 2, 3, 5, 3, 8, 4, 7, 5, 6, 7, 6, 8, 7, 8 /)
    A = (/ 7.0_wp, 1.0_wp, 2.0_wp, 7.0_wp, -4.0_wp, 8.0_wp, 2.0_wp, 1.0_wp,    &
           5.0_wp, 7.0_wp, 9.0_wp, 5.0_wp, 1.0_wp, 5.0_wp,-1.0_wp, 5.0_wp,     &
           11.0_wp, 5.0_wp /)
  END IF

!  set the RHS so that the required solution is a vector of ones

! B = 1.0_wp ! set right-hand side
  B = 0.0_wp
  DO i = 1, n
    DO l = IA( i ), IA( i + 1 ) - 1
      j = JA( l )
      B( i, 1 ) = B( i, 1 ) + A( l )
      IF ( i /= j ) B( j, 1 ) = B( j, 1 ) + A( l )
    END DO
  END DO

!  set defaults

  CALL WSETMAXTHRDS( num_threads )
  CALL WSMP_initialize( )
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
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, bdum, 1, 1, aux, naux, MRP,   &
                IPARM, DPARM )

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
    CALL WSSMP( n, IA, JA, A, DIAG, PERM, INVP, bdum, 1, 1, aux, naux, MRP,    &
                IPARM, DPARM )

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
  CALL WSMP_clear( )
  CALL WSSFREE( )

END PROGRAM TEST_WSMP
