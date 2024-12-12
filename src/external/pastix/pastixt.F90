! Fortran 90 PaStiX example using a 5 by 5 symmetric matrix in different formats
! Example by Mathieu Faverge, modified by Nick Gould, this version 2024-06-11

#include "galahad_modules.h"

PROGRAM test_pastix
  USE GALAHAD_KINDS_precision
  USE iso_c_binding
  USE spmf_enums
  USE spmf_interfaces_precision
  USE pastixf_enums, MPI_COMM_WORLD_pastix_duplic8 => MPI_COMM_WORLD
  USE pastixf_interfaces_precision
! USE spmf
! USE pastixf
  IMPLICIT NONE
  INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
  INTEGER ( KIND = ip_ ), PARAMETER :: ne = 7
  INTEGER ( KIND = ip_ ), PARAMETER :: neu = 11
  INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: ROW, COL, PTR
  REAL ( KIND = rp_ ), POINTER, DIMENSION( : ) :: VAL
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( :, : ) :: x0, x, b
  TYPE( pastix_data_t ), POINTER :: pastix_data
  TYPE( spmatrix_t ), POINTER :: spm, spm2
  INTEGER ( kind = pastix_int_t ), target :: iparm( iparm_size )
  REAL ( kind = c_precision ), target :: dparm( dparm_size )
  INTEGER ( KIND = ip_ ) :: nrhs, store
  INTEGER ( ipc_ ) :: info
  INTEGER ( kind = pastix_int_t ), DIMENSION( : ), POINTER :: permtab
  TYPE ( pastix_order_t ), POINTER :: order => NULL( )
#ifdef REAL_32
  REAL ( kind = dp_ ) :: eps = 0.001_dp_
#elif REAL_128
  REAL ( kind = dp_ ) :: eps = 0.000000000001_dp_
#else
  REAL ( kind = dp_ ) :: eps = 0.000000000001_dp_
#endif

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
#ifdef REAL_32
      spm%flttype = SpmFloat
#else
      spm%flttype = SpmDouble
#endif
      spm%fmttype = SpmIJV
      spm%n = n
      spm%nnz = ne
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
#ifdef REAL_32
write(6,"('single')")
      CALL spmGetArray( spm, colptr = COL, rowptr = ROW, svalues = VAL )
#elif REAL_128
      CALL spmGetArray( spm, colptr = COL, rowptr = ROW, qvalues = VAL )
#else
write(6,"('double')")
      CALL spmGetArray( spm, colptr = COL, rowptr = ROW, dvalues = VAL )
#endif

!   set the matrix

write(6,*) ' size of row, col, val = ', SIZE( ROW ), SIZE( COL ), SIZE( VAL )
      ROW( : ne ) = (/ 1, 2, 3, 3, 4, 5, 5 /)
      COL( : ne ) = (/ 1, 1, 2, 3, 3, 2, 5 /)
      VAL( : ne ) = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_, 6.0_rp_,   &
                       1.0_rp_ /)
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
#ifdef REAL_32
      spm%flttype = SpmFloat
#else
      spm%flttype = SpmDouble
#endif
      spm%fmttype = SpmCSC
      spm%n = n
      spm%nnz = ne
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
#ifdef REAL_32
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, svalues = VAL )
#elif REAL_128
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, qvalues = VAL )
#else
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, dvalues = VAL )
#endif

!   set the matrix

      PTR( : n + 1 ) = (/ 1, 3, 5, 7, 7, 8 /)
      ROW( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
      VAL( : ne ) = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 6.0_rp_, 1.0_rp_, 5.0_rp_,   &
                       1.0_rp_ /)

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
#ifdef REAL_32
      spm%flttype = SpmFloat
#else
      spm%flttype = SpmDouble
#endif
      spm%fmttype = SpmCSC
      spm%n = n
      spm%nnz = neu
      spm%dof = 1
      CALL spmUpdateComputedFields( spm )
      CALL spmAlloc( spm )
#ifdef REAL_32
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, svalues = VAL )
#elif REAL_128
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, qvalues = VAL )
#else
      CALL spmGetArray( spm, colptr = PTR, rowptr = ROW, dvalues = VAL )
#endif

!   set the matrix

      PTR( : n + 1 ) = (/ 1, 3, 6, 9, 10, 12 /)
      ROW( : neu ) = (/ 1, 2, 1, 3, 5, 2, 3, 4, 3, 2, 5 /)
      VAL( : neu ) = (/ 2.0_rp_, 3.0_rp_, 3.0_rp_, 4.0_rp_, 6.0_rp_, 4.0_rp_,  &
                        1.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 1.0_rp_ /)

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
    X0( : n, 1 ) = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_ /)
    B( : n, 1 ) = (/ 8.0_rp_, 45.0_rp_, 31.0_rp_, 15.0_rp_, 17.0_rp_ /)
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
    CALL pastix_task_solve( pastix_data, n, nrhs, X, spm%nexp, info )

! Refine the solution
    CALL pastix_task_refine( pastix_data, spm%nexp, nrhs, B, spm%nexp, X,      &
                             spm%nexp, info )

!   Destroy the C data structure
    call pastixFinalize( pastix_data )

!   Check the solution

    CALL spmCheckAxb( eps, nrhs, spm, X0, spm%nexp, B, spm%nexp, X, spm%nexp, &
                      info )

    CALL spmExit( spm )
    DEALLOCATE( spm )
    DEALLOCATE( x0, x, b )

  END DO

END PROGRAM test_pastix
