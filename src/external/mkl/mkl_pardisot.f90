PROGRAM TEST_PARDISO

  USE MKL_PARDISO
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE

!  precision

  INTEGER, PARAMETER :: wp = KIND( 1.0D0 )

!  internal solver memory pointer

  TYPE ( MKL_PARDISO_HANDLE ), ALLOCATABLE, DIMENSION( : ) :: PT

!  all other variables

  INTEGER, PARAMETER :: n = 8
  INTEGER:: nz, maxfct, mnum, mtype, phase, nrhs, error, msglvl
  INTEGER, ALLOCATABLE, DIMENSION ( : ) :: IPARM, IA, JA
  REAL( KIND = wp ), ALLOCATABLE, DIMENSION ( : ) :: A, B, X
  INTEGER :: i, j, l, idum( 1 )
  REAL( KIND = wp ) :: ddum( 1 )
  LOGICAL :: all_in_one = .FALSE.
  LOGICAL :: easy_problem = .FALSE.

  ALLOCATE( IPARM( 64 ), PT( 64 ) )
  ALLOCATE( IA( n + 1 ), B( n ), X( n ) )

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
  nrhs = 1

!  set the RHS so that the required solution is a vector of ones

! B = 1.0_wp ! set right-hand side
  B = 0.0_wp
  DO i = 1, n
    DO l = IA( i ), IA( i + 1 ) - 1
      j = JA( l )
      B( i ) = B( i ) + A( l )
      IF ( i /= j ) B( j ) = B( j ) + A( l )
    END DO
  END DO

!  set defaults

  mtype = - 2 ! symmetric, indefinite
! mtype = 1 ! symmetric
! mtype = 2 ! symmetric, positive definite

  DO i = 1, 64
    IPARM( i ) = 0
    PT( i )%DUMMY = 0 
  END DO

!  set up PARDISO control parameters

  IPARM = 0  ! this sets all input parameters to defaul values
! IPARM( 1 ) = 1 ! no solver default
! IPARM( 2 ) = 0 ! minimum degree
!!IPARM( 2 ) = 2 ! fill-in reordering from METIS
! IPARM( 3 ) = 1 ! numbers of processors
! IPARM( 4 ) = 0 ! no iterative-direct algorithm
! IPARM( 5 ) = 0 ! no user fill-in reducing permutation
! IPARM( 6 ) = 0 ! =0 solution on the first n compoments of x
! IPARM( 7 ) = 0 ! not in use
! IPARM( 8 ) = 2 ! numbers of iterative refinement steps
! IPARM( 9 ) = 0 ! not in use
! IPARM( 10 ) = 13 ! perturbe the pivot elements with 1E-13
! IPARM( 11 ) = 1 ! use nonsymmetric permutation and scaling MPS
! IPARM( 12 ) = 0 ! not in use
! IPARM( 13 ) = 0 ! maximum weighted matching algorithm is switched-off
! IPARM( 14 ) = 0 ! Output: number of perturbed pivots
! IPARM( 15 ) = 0 ! not in use
! IPARM( 16 ) = 0 ! not in use
! IPARM( 17 ) = 0 ! not in use
! IPARM( 18 ) = -1 ! Output: number of nonzeros in the factor LU
! IPARM( 19 ) = -1 ! Output: Mflops for LU factorization
! IPARM( 20 ) = 0 ! Output: Numbers of CG Iterations
! IPARM( 27 ) = 1 ! check matrix

  maxfct = 1
  mnum = 1

  error = 0 ! initialize error flag
  msglvl = 0 ! don't print statistical information
! msglvl = 1 ! print statistical information

!  Either solve the problem in a single phase ...

  IF ( all_in_one ) THEN
    phase = 13 ! complete solution
    CALL MKL_PARDISO_SOLVE( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,      &
                            idum, nrhs, IPARM, msglvl, B, X, error )
    IF ( error == GALAHAD_unavailable_option ) THEN
      WRITE( 6, "( ' MKL PARDISO is not available' )" )
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
    CALL MKL_PARDISO_SOLVE( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,      &
                            idum, nrhs, IPARM, msglvl, ddum, ddum, error )

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
    CALL MKL_PARDISO_SOLVE( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,      &
                            idum, nrhs, IPARM, msglvl, ddum, ddum, error )

    WRITE( 6, "( ' factorization completed ... ' )" )
    IF ( error /= 0 ) THEN
      WRITE( 6, "( ' the following ERROR was detected: ', I0 )" ) error
      GO TO 1
    END IF

!  back substitution and iterative refinement

    iparm(8) = 2 ! max numbers of iterative refinement steps
    phase = 33 ! only solution
    CALL MKL_PARDISO_SOLVE( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,      &
                            idum, nrhs, IPARM, msglvl, B, X, error )

    WRITE( 6, * ) ' solve completed ... '
  END IF

  WRITE( 6, "( ' the solution of the system is X =' )" )
  WRITE( 6, "( ( 4ES12.4 ) )" ) X( : n )

!  termination and release of memory

1 CONTINUE 
  phase = - 1 ! release internal memory
  CALL MKL_PARDISO_SOLVE( PT, maxfct, mnum, mtype, phase, n, ddum, idum, idum, &
                          idum, nrhs, IPARM, msglvl, ddum, ddum, error )
  DEALLOCATE( IPARM, IA, JA, A, B, X, PT )

END PROGRAM TEST_PARDISO
