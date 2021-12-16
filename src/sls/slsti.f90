! THIS VERSION: GALAHAD 3.3 - 01/12/2021 AT 16:30 GMT.
   PROGRAM GALAHAD_SLS_interface_test
   USE GALAHAD_SLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_full_data_type ) :: data
   TYPE ( SLS_control_type ) control
   TYPE ( SLS_inform_type ) :: inform
   INTEGER :: i, l, ordering, scaling, solver, storage_type, s, status
   INTEGER, PARAMETER :: n = 5, ne  = 7
   REAL ( KIND = wp ), PARAMETER :: good_x = EPSILON( 1.0_wp ) ** 0.333
   REAL ( KIND = wp ), DIMENSION( n ) :: X
   INTEGER, DIMENSION( 0 ) :: null
   INTEGER, DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER, DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 5 /)
   INTEGER, DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = wp ), DIMENSION( ne ) ::                                      &
     val = (/ 2.0_wp, 3.0_wp, 6.0_wp, 4.0_wp, 1.0_wp, 5.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n * ( n + 1 ) / 2 ) ::                       &
     dense = (/ 2.0_wp, 3.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp, 0.0_wp,        &
                0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp, 6.0_wp, 0.0_wp, 0.0_wp,        &
                1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     rhs = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     SOL = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)

!  =====================================
!  basic test of various storage formats
!  =====================================

! assign the matrix
   WRITE( 6, "( ' storage         RHS   refine   partial')" )
   DO storage_type = 1, 3
     CALL SLS_initialize( 'sils', data, control, inform )
     IF ( inform%status < 0 ) THEN
       CALL SLS_information( data, inform, status )
       WRITE( 6, "( '  fail in initialize, status = ', i0 )", advance = 'no' ) &
         inform%status
       WRITE( 6, "( '' )" )
       CYCLE
     END IF
! analyse the matrix structure
     SELECT CASE( storage_type )
     CASE ( 1 )
       WRITE( 6, "( A15 )", advance = 'no' ) " coordinate    "
       CALL SLS_analyse_matrix( control, data, status, n,                      &
                                'coordinate', ne, row, col, null )
     CASE ( 2 )
       WRITE( 6, "( A15 )", advance = 'no' ) " sparse by rows"
       CALL SLS_analyse_matrix( control, data, status, n,                      &
                                'sparse_by_rows', ne, null, col, ptr )
     CASE ( 3 )
       WRITE( 6, "( A15 )", advance = 'no' ) " dense         "
       CALL SLS_analyse_matrix( control, data, status, n,                      &
                                'dense', ne, null, null, null )
     END SELECT
     IF ( status < 0 ) THEN
       CALL SLS_information( data, inform, status )
       WRITE( 6, "( '  fail in analyse, status = ', I0 )", advance = 'no' )    &
         inform%status
       CYCLE
     END IF
! factorize the matrix
     IF ( storage_type == 3 ) THEN
       CALL SLS_factorize_matrix( data, status, dense )
     ELSE
       CALL SLS_factorize_matrix( data, status, val )
     END IF
     IF ( inform%status < 0 ) THEN
       CALL SLS_information( data, inform, status )
       WRITE( 6, "( '  fail in factorize, status = ', i0 )", advance = 'no' )  &
         inform%status
       CYCLE
     END IF
! solve without refinement
     control%max_iterative_refinements = 0
     X = rhs
     CALL SLS_solve_system( data, status, X )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) ) <= good_x ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF
! perform one refinement
     control%max_iterative_refinements = 1
     CALL SLS_reset_control( control, data, status )
     X = rhs
     CALL SLS_solve_system( data, status, X )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) ) <= good_x ) THEN
       WRITE( 6, "( '    ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail  ' )", advance = 'no' )
     END IF
     CALL SLS_information( data, inform, status )
! obtain solution by part solves
     X = rhs
     CALL SLS_partial_solve( 'L', data, status, X )
     IF ( status < 0 ) THEN
       CALL SLS_information( data, inform, status )
       WRITE( 6, "( '    fail ' )" )
       CYCLE
     END IF
     CALL SLS_partial_solve( 'D', data, status, X )
     CALL SLS_partial_solve( 'U', data, status, X )
     CALL SLS_information( data, inform, status )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) ) <= good_x ) THEN
       WRITE( 6, "( '     ok  ' )" )
     ELSE
       WRITE( 6, "( '    fail ' )" )
     END IF
     CALL SLS_terminate( data, control, inform )
   END DO

   STOP
   END PROGRAM GALAHAD_SLS_interface_test
