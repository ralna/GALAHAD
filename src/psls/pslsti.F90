! THIS VERSION: GALAHAD 5.0 - 2024-06-06 AT 13:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_PSLS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_PSLS_precision
   IMPLICIT NONE
   TYPE ( PSLS_full_data_type ) :: data
   TYPE ( PSLS_control_type ) control
   TYPE ( PSLS_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: storage_type, s, status, status_sol
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, ne  = 7
   REAL ( KIND = rp_ ), PARAMETER :: good_x = EPSILON( 1.0_rp_ ) ** 0.333
   REAL ( KIND = rp_ ), DIMENSION( n ) :: X
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = rp_ ), DIMENSION( ne ) ::                                     &
     val = (/ 2.0_rp_, 3.0_rp_, 6.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n * ( n + 1 ) / 2 ) ::                      &
     dense = (/ 2.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 4.0_rp_, 1.0_rp_, 0.0_rp_, &
                0.0_rp_, 5.0_rp_, 0.0_rp_, 0.0_rp_, 6.0_rp_, 0.0_rp_, 0.0_rp_, &
                1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     rhs = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
!  =====================================
!  basic test of various storage formats
!  =====================================
!  loop over storage types; select the preconditioner
   DO storage_type = 1, 3
     CALL PSLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%print_level = 10
     control%preconditioner = 2  ! banded preconditioner
     control%semi_bandwidth = 1  ! semi-bandwidth of one
!    control%definite_linear_solver = 'sils'
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

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( PSLS_control_type ) :: control
#include "galahad_sls_defaults.h"
     control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_PSLS_interface_test

