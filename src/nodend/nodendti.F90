! THIS VERSION: GALAHAD 5.2 - 2025-03-303 AT 10:55 GMT.

#include "galahad_modules.h"

     PROGRAM NODEND_interface_test
     USE GALAHAD_KINDS, ONLY: ip_
     USE GALAHAD_NODEND_precision
     USE GALAHAD_SMT_precision
     INTEGER, PARAMETER :: out = 6
     INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, A_ne = 4
     INTEGER ( KIND = ip_ ), DIMENSION( A_ne ) :: A_row = (/ 1, 2, 3, 3 /)
     INTEGER ( KIND = ip_ ), DIMENSION( A_ne ) :: A_col = (/ 1, 2, 1, 3 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: A_ptr = (/ 1, 2, 3, 5 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM
     TYPE ( NODEND_control_type ) :: control
     TYPE ( NODEND_inform_type ) :: inform
     TYPE ( NODEND_full_data_type ) :: data
     INTEGER ( KIND = ip_ ) :: type, status

!  test the different storage versions, fortran indexing

     WRITE( out, "( ' fortran indexing', / )" )
     DO type = 1, 3
       CALL NODEND_full_initialize( data, control, inform )
       SELECT CASE( type )
       CASE( 1 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'COORDINATE', A_ne, A_row, A_col, A_ptr )
       CASE( 2 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'SPARSE_BY_ROWS', A_ne, A_row, A_col, A_ptr )
       CASE( 3 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'DENSE',  A_ne, A_row, A_col, A_ptr )
       END SELECT
       CALL NODEND_information( data, inform, status )
       IF ( PERM( 1 ) <= 0 ) THEN
         WRITE( out, "( ' No Nodend ', A, ' available, stopping' )" )          &
           TRIM( control%version )
       ELSE IF ( inform%status < 0 ) THEN
         WRITE( out, "( ' Nodend type ', I0,                                   &
        &  ' storage failure, status = ', I0 )" ) type, inform%status
       ELSE 
         IF ( inform%status == 0 ) THEN
           WRITE( out, "(  ' Nodend type ', I0,                                &
          &  ' storage call successful, permutation = ', 3( I2 ) )" ) type, PERM
         ELSE
           WRITE( out, "(  ' Nodend type ', I0,                                &
          & ' storage call unsuccessful, no permutation found' )" ) type
         END IF
       END IF
     END DO

!  test the different storage versions, C indexing

     WRITE( out, "( /, ' C indexing', / )" )
     A_row = A_row - 1 ; A_col = A_col - 1 ; A_ptr = A_ptr - 1
     DO type = 1, 3
       CALL NODEND_full_initialize( data, control, inform )
       data%f_indexing = .FALSE.
       SELECT CASE( type )
       CASE( 1 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'COORDINATE', A_ne, A_row, A_col, A_ptr )
       CASE( 2 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'SPARSE_BY_ROWS', A_ne, A_row, A_col, A_ptr )
       CASE( 3 )
         CALL NODEND_order_a( control, data, status, n, PERM,                  &
                              'DENSE',  A_ne, A_row, A_col, A_ptr )
       END SELECT
       CALL NODEND_information( data, inform, status )
       IF ( PERM( 1 ) < 0 ) THEN
         WRITE( out, "( ' No Nodend ', A, ' available, stopping' )" )          &
           TRIM( control%version )
       ELSE IF ( inform%status < 0 ) THEN
         WRITE( out, "( ' Nodend type ', I0,                                   &
        &  ' storage failure, status = ', I0 )" ) type, inform%status
       ELSE 
         IF ( inform%status == 0 ) THEN
           WRITE( out, "(  ' Nodend type ', I0,                                &
          &  ' storage call successful, permutation = ', 3( I2 ) )" ) type, PERM
         ELSE
           WRITE( out, "(  ' Nodend type ', I0,                                &
          & ' storage call unsuccessful, no permutation found' )" ) type
         END IF
       END IF
     END DO

     END PROGRAM NODEND_interface_test
