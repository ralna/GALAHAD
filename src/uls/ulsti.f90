! THIS VERSION: GALAHAD 3.3 - 09/12/2021 AT 10:030 GMT.
   PROGRAM GALAHAD_ULS_interface_test
   USE GALAHAD_SYMBOLS
   USE GALAHAD_ULS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( ULS_full_data_type ) :: data
   TYPE ( ULS_control_type ) control
   TYPE ( ULS_inform_type ) :: inform
   INTEGER, DIMENSION( 0 ) :: null
   INTEGER :: i, storage_type, status
   INTEGER, PARAMETER :: m = 5, n = 5, ne  = 7
   REAL ( KIND = wp ) :: B( n ), X( n )
   INTEGER, DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER, DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 4 /)
   INTEGER, DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = wp ), DIMENSION( ne ) ::                                      &
     val = (/ 2.0_wp, 3.0_wp, 6.0_wp, 4.0_wp,  1.0_wp, 5.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n * n ) ::                                   &
     dense = (/ 2.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                        &
                3.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 6.0_wp,                        &
                0.0_wp, 4.0_wp, 1.0_wp, 0.0_wp, 0.0_wp,                        &
                0.0_wp, 0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp,                        &
                0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 0.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     rhs = (/ 2.0_wp, 33.0_wp, 11.0_wp, 15.0_wp, 4.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     rhst = (/ 8.0_wp, 12.0_wp, 23.0_wp, 5.0_wp, 12.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     SOL = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)

   WRITE( 6,"( ' storage         RHS   refine   RHST  refine')")
! read matrix order and number of entries
!  DO storage_type = 1, 3
   DO storage_type = 3, 3
! initialize the structures for the gls solver
     CALL ULS_initialize( 'gls ', data, control, inform )
     IF ( inform%status < 0 ) THEN
       CALL ULS_information( data, inform, status )
       WRITE( 6, "( '  fail in initialize, status = ', i0 )", advance = 'no' ) &
         inform%status
       WRITE( 6, "( '' )" )
       CYCLE
     END IF
! factorize! analyse the matrix structure
     SELECT CASE( storage_type )
     CASE ( 1 )
       WRITE( 6, "( A15 )", advance = 'no' ) " coordinate    "
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'coordinate', ne, val, row, col, null )
     CASE ( 2 )
       WRITE( 6, "( A15 )", advance = 'no' ) " sparse by rows"
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'sparse_by_rows', ne, val, null, col, ptr )
     CASE ( 3 )
       WRITE( 6, "( A15 )", advance = 'no' ) " dense         "
       CALL ULS_factorize_matrix( control, data, status, m, n,                 &
                                  'dense', ne, dense, null, null, null )
     END SELECT
     IF ( status < 0 ) THEN
       CALL ULS_information( data, inform, status )
       WRITE( 6, "( '  fail in analyse, status = ', I0 )", advance = 'no' )    &
         inform%status
       CYCLE
     END IF
! solve without refinement
     X = rhs
     CALL ULS_solve_system( data, status, X, .FALSE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF
! perform one refinement
     control%max_iterative_refinements = 1
     CALL ULS_reset_control( control, data, status )
     X = rhs
     CALL ULS_solve_system( data, status, X, .FALSE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
       WRITE( 6, "( '    ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail  ' )", advance = 'no' )
     END IF
! solve transpose system without refinement
     X = rhst
     CALL ULS_solve_system( data, status, X, .TRUE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
       WRITE( 6, "( '   ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '  fail ' )", advance = 'no' )
     END IF
! perform one refinement
     control%max_iterative_refinements = 1
     CALL ULS_reset_control( control, data, status )
     X = rhst
     CALL ULS_solve_system( data, status, X, .TRUE. )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                            &
             <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
       WRITE( 6, "( '    ok  ' )" )
     ELSE
       WRITE( 6, "( '  fail  ' )" )
     END IF
     CALL ULS_terminate( data, control, inform )
   END DO
   STOP
   END PROGRAM GALAHAD_ULS_interface_test
