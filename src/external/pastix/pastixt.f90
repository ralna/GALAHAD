! Fortran 90 PaStiX example using a 5 by 5 symmetric matrix in different formats
! Example by Mathieu Faverge, modified by Nick Gould
! 2022-10-18

PROGRAM test_pastix
  USE iso_c_binding
  USE spmf
  USE pastixf
  IMPLICIT NONE
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, PARAMETER :: n = 5
  INTEGER, PARAMETER :: ne = 7
  INTEGER, PARAMETER :: neu = 11
  INTEGER, POINTER, DIMENSION( : ) :: ROW, COL, PTR
  REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: VAL
  REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: x0, x, b
  TYPE( pastix_data_t ), POINTER :: pastix_data
  TYPE( spmatrix_t ), POINTER :: spm, spm2
  INTEGER ( kind = pastix_int_t ), target :: iparm( iparm_size )
  REAL ( kind = c_double ), target :: dparm( dparm_size )
  INTEGER :: nrhs, store
  INTEGER( c_int ) :: info
  INTEGER ( kind = pastix_int_t ), DIMENSION( : ), POINTER :: permtab
  TYPE ( pastix_order_t ), POINTER :: order => NULL( )

  DO store = 1, 3

! Initialize the parameters and the solver
! (Done before any calls to spm to automatically intialize MPI if needed)

    CALL pastixInitParam( iparm, dparm )
    iparm( 1 ) = 0
    iparm( 6 ) = 0
    IF ( store == 3 ) THEN
      iparm( 44 ) = 2
    ELSE
      iparm( 44 ) = 1
    END IF
    CALL pastixInit( pastix_data, MPI_COMM_WORLD, iparm, dparm )

!   Create the spm out of the internal data

    ALLOCATE( spm )
    CALL spmInit( spm )
    IF ( spm%mtxtype == - 1 ) THEN
      WRITE( 6, * ) ' no PaStiX available'
      STOP
    END IF

!  store the matrix in coordinate form

    IF ( store == 1 ) THEN
      spm%baseval = 1
      spm%mtxtype = SpmSymmetric
      spm%flttype = SpmDouble
      spm%fmttype = SpmIJV
      spm%n = n
      spm%nnz = ne
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
      CALL spmGetArray( spm, colptr = COL, rowptr = ROW, dvalues = VAL )

!   set the matrix

      ROW( : ne ) = (/ 1, 2, 3, 3, 4, 5, 5 /)
      COL( : ne ) = (/ 1, 1, 2, 3, 3, 2, 5 /)
      VAL( : ne ) = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp, 6.0_wp, 1.0_wp /)

      IF ( .TRUE. ) THEN
      ALLOCATE( spm2 )
      CALL spmCheckAndCorrect( spm, spm2, info )
      IF ( info /= 0 ) THEN
         CALL spmExit( spm )
         spm = spm2
      END IF
      DEALLOCATE( spm2 )
      END IF

!  store the matrix in CSC form

    ELSE IF ( store == 2 ) THEN
      spm%baseval = 1
      spm%mtxtype = SpmSymmetric
      spm%flttype = SpmDouble
      spm%fmttype = SpmCSC
      spm%n = n
      spm%nnz = ne
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, dvalues = VAL )

!   set the matrix

      PTR( : n + 1 ) = (/ 1, 3, 5, 7, 7, 8 /)
      ROW( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
      VAL( : ne ) = (/ 2.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 1.0_wp, 5.0_wp, 1.0_wp /)

!     IF ( .TRUE. ) THEN
      IF ( .FALSE. ) THEN
      ALLOCATE( spm2 )
      CALL spmCheckAndCorrect( spm, spm2, info )
      IF ( info /= 0 ) THEN
         CALL spmExit( spm )
         spm = spm2
      END IF
      DEALLOCATE( spm2 )
      END IF

!  store the matrix as general in CSC form

    ELSE IF ( store == 3 ) THEN
      spm%baseval = 1
      spm%mtxtype = SpmGeneral
      spm%flttype = SpmDouble
      spm%fmttype = SpmCSC
      spm%n = n
      spm%nnz = neu
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, dvalues = VAL )

!   set the matrix

      PTR( : n + 1 ) = (/ 1, 3, 6, 9, 10, 12 /)
      ROW( : neu ) = (/ 1, 2, 1, 3, 5, 2, 3, 4, 3, 2, 5 /)
      VAL( : neu ) = (/ 2.0_wp, 3.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 4.0_wp,        &
                        1.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 1.0_wp /)

!     IF ( .TRUE. ) THEN
      IF ( .FALSE. ) THEN
      ALLOCATE( spm2 )
      CALL spmCheckAndCorrect( spm, spm2, info )
      IF ( info /= 0 ) THEN
         CALL spmExit( spm )
         spm = spm2
      END IF
      DEALLOCATE( spm2 )
      END IF
    END IF

!   CALL spmPrintInfo( spm )

!    set the right hand side for a solution of ( i, i = 1, n )

    nrhs = 1
    ALLOCATE( x0(spm%nexp, nrhs), x( spm%nexp, nrhs), b( spm%nexp, nrhs) )
!   call spmGenRHS( SpmRhsRndX, nrhs, spm, x0, spm%nexp, b, spm%nexp, info )
    X0( : n, 1 ) = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)
    B( : n, 1 ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
    X = B

!   Solve the problem

!   Analyze the problem
    CALL pastix_task_analyze( pastix_data, spm, info )

!   Factorize the matrix
    CALL pastix_task_numfact( pastix_data, spm, info )
    write(6,*) ' info ', info

! If needed, get the generated ordering
    CALL pastixOrderGet( pastix_data, order )

! Convert the permtab to Fortran array
    CALL pastixOrderGetArray( order, permtab = permtab )
    write(6, *) ' order ', permtab( 1 : n ) + 1

! Solve the problem
    CALL pastix_task_solve( pastix_data, nrhs, X, spm%nexp, info )

! Refine the solution
    CALL pastix_task_refine( pastix_data, spm%nexp, nrhs, B, spm%nexp, X,      &
                             spm%nexp, info )

!   Destroy the C data structure
    call pastixFinalize( pastix_data )

!   Check the solution

    CALL spmCheckAxb( dparm( DPARM_EPSILON_REFINEMENT ), nrhs, spm, X0,        &
                      spm%nexp, B, spm%nexp, X, spm%nexp, info )

    CALL spmExit( spm )
    DEALLOCATE( spm )
    DEALLOCATE( x0, x, b )

  END DO

END PROGRAM test_pastix
