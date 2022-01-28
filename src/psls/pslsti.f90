   PROGRAM GALAHAD_PSLS_interface_test !  GALAHAD 4.0 - 2022-01-25 AT 09:35 GMT.
   USE GALAHAD_PSLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( PSLS_full_data_type ) :: data
   TYPE ( PSLS_control_type ) control
   TYPE ( PSLS_inform_type ) :: inform
   INTEGER :: storage_type, s, status, status_sol
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

!  =====================================
!  basic test of various storage formats
!  =====================================

!  loop over storage types; select the preconditioner
   DO storage_type = 1, 3
     CALL PSLS_initialize( data, control, inform )
      control%print_level = 10
      control%preconditioner = 2  ! banded preconditioner
      control%semi_bandwidth = 1  ! semi-bandwidth of one
      control%definite_linear_solver = 'sils'
! import the matrix structure
     SELECT CASE( storage_type )
     CASE ( 1 )
       WRITE( 6, "( A15 )", advance = 'no' ) " coordinate    "
       CALL PSLS_import( control, data, status, n,                             &
                         'coordinate', ne, row, col, null )
     CASE ( 2 )
       WRITE( 6, "( A15 )", advance = 'no' ) " sparse by rows"
       CALL PSLS_import( control, data, status, n,                             &
                         'sparse_by_rows', ne, null, col, ptr )
     CASE ( 3 )
       WRITE( 6, "( A15 )", advance = 'no' ) " dense         "
       CALL PSLS_import( control, data, status, n,                             &
                         'dense', ne, null, null, null )
     END SELECT
     IF ( status < 0 ) THEN
       CALL PSLS_information( data, inform, status )
       WRITE( 6, "( '  fail in import, status = ', I0 )", advance = 'no' )     &
         inform%status
       CYCLE
     END IF
! form and factorize the preconditioner
     IF ( storage_type == 3 ) THEN
       CALL PSLS_form_preconditioner( data, status, dense )
     ELSE
       CALL PSLS_form_preconditioner( data, status, val )
     END IF
! solve without refinement
     IF ( status == 0 ) THEN
       CALL PSLS_information( data, inform, status )
       X = rhs
       CALL PSLS_apply_preconditioner( data, status_sol, X )
     ELSE
       status_sol = - 1
     END IF
     WRITE( 6, "( ' storage: status form & factorize = ', I2,                  &
    &           ' solve = ', I2 )" ) status, status_sol
! clean up
     CALL PSLS_terminate( data, control, inform )
   END DO

   STOP
   END PROGRAM GALAHAD_PSLS_interface_test

