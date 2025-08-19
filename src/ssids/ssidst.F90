! THIS VERSION: GALAHAD 5.3 - 2025-08-15 AT 11:50 GMT

#include "spral_procedures.h"

   PROGRAM GALAHAD_SSIDS_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SSIDS_precision
   IMPLICIT NONE
   TYPE( ssids_inform_type ) :: inform
   TYPE( ssids_akeep_type ) :: akeep
   TYPE( ssids_fkeep_type ) :: fkeep
   TYPE( ssids_control_type ) :: control
   INTEGER ( KIND = ip_ ) :: i, ordering, cuda_error
!  LOGICAL :: mpi_flag
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, ne  = 7
   INTEGER ( KIND = ip_ ) :: ORDER( n )
   REAL ( KIND = rp_ ) :: X( n ), X2( n, 2 ), D( 2, n )
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: ROW = (/ 1, 2, 3, 5, 3, 4, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: PTR = (/ 1, 3, 5, 7, 7, 8 /)
   REAL ( KIND = rp_ ), DIMENSION( ne ) ::                                     &
     VAL = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 6.0_rp_, 1.0_rp_, 5.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     RHS = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     SOL = (/ 1.0_rp_,  2.0_rp_,  3.0_rp_,  4.0_rp_,  5.0_rp_ /)

#ifdef REAL_32
   WRITE( 6, "( '  ssids single precision tests', / )" )
#elif REAL_128
   WRITE( 6, "( '  ssids quadruple precision tests', / )" )
#else
   WRITE( 6, "( '  ssids double precision tests', / )" )
#endif

!  test external ordering strategies

!  DO ordering = 0, 1 ! all orderings
   DO ordering = 1, 1 ! provided ordering
     IF ( ordering == 0 ) THEN
       WRITE( 6, "( '  default ordering' )" )
     ELSE
       WRITE( 6, "( '  provided ordering' )" )
     END IF
     WRITE( 6, "( '  RHS    refine  partial')" )

! analyse the structure

!    WRITE( 6, * ) ' analyse '
!    control%print_level = 100
     IF ( ordering == 1 ) THEN
       DO i = 1, n ; ORDER( i ) = n - i + 1 ; END DO
       control%ordering = 0
       CALL ssids_analyse( .FALSE., n, PTR, ROW, akeep, control, inform,       &
                           val = VAL, order = ORDER )
     ELSE
       CALL ssids_analyse( .FALSE., n, PTR, ROW, akeep, control, inform,       &
                           val = VAL )
     END IF
     IF ( inform%flag < 0 ) THEN
       WRITE( 6, "( '  fail in analyse, status = ', i0 )",                     &
              advance = 'no' ) inform%flag
       WRITE( 6, "( '' )" )
       CYCLE
     END IF

! factorize the matrix

!write(6,*) ' factorize '
     CALL ssids_factor( .FALSE., VAL, akeep, fkeep, control, inform,           &
                        ptr = PTR, row = ROW )
     IF ( inform%flag < 0 ) THEN
       WRITE( 6, "( '  fail in factorize, status = ', I0 )",                   &
              advance = 'no' ) inform%flag
       WRITE( 6, "( '' )" )
       CYCLE
     END IF

! assign the right-hand side, and solve without refinement

!write(6,*) ' solve 1 RHS'
     X = RHS
     CALL ssids_solve( X, akeep, fkeep, control, inform )
!write(6,"( ' X = ', 5ES10.2 )" ) X( 1 : n )
!write(6,*) ' status - ', inform%flag

     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
!write(6,*) MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF

! Solve multiple RHS without refinement

     X2( : , 1 ) = RHS ; X2( : , 2 ) = RHS
     CALL ssids_solve( 2_ip_, X2, n, akeep, fkeep, control, inform )

!write(6,*) ' status - ', inform%flag
     IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                       &
             <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                                &
          MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                       &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '      ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '     fail ' )", advance = 'no' )
     END IF
! Obtain solution by part solves

     X = RHS
!write(6,*) ' L '
     CALL ssids_solve( X, akeep, fkeep, control, inform, job = 1_ip_ )
     IF (inform%flag /= 0 ) THEN
       WRITE( 6, "( '    fail ' )", advance = 'no' )
       WRITE( 6, "( '' )" )
       CYCLE
     END IF
!write(6,*) ' D '
     CALL ssids_solve( X, akeep, fkeep, control, inform, job = 2_ip_ )
!write(6,*) ' U '
     CALL ssids_solve( X, akeep, fkeep, control, inform, job = 3_ip_ )
!write(6,*) ' E '
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_rp_ ) ** 0.333 ) THEN
       WRITE( 6, "( '     ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '    fail ' )", advance = 'no' )
     END IF

!  enquire about factors and modify diagonals

!write(6,*) ' enquire '
     CALL ssids_enquire_indef( akeep, fkeep, control, inform,                  &
                               piv_order = ORDER, d = D )
!write(6,*) ' alter d '
     CALL ssids_alter( D, akeep, fkeep, control, inform)

!  free data

!write(6,*) ' terminate '
     CALL ssids_free(akeep, fkeep, cuda_error )
     WRITE( 6, "( '' )" )
   END DO
!  CALL MPI_INITIALIZED( mpi_flag, i )
!  IF ( mpi_flag ) CALL MPI_FINALIZE( i )
   WRITE( 6, "( /, '  ssids tests completed' )" )
   STOP
   END PROGRAM GALAHAD_SSIDS_test_program
