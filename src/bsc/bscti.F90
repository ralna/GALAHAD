! THIS VERSION: GALAHAD 5.3 - 2025-05-25 AT 11:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BSC_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BSC_precision
   IMPLICIT NONE
   TYPE ( BSC_full_data_type ) :: data
   TYPE ( BSC_control_type ) :: control
   TYPE ( BSC_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 3, n = 4, a_ne = 6
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: S_row, S_col, S_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S_val
   CHARACTER ( LEN = 15 ) :: A_type
   REAL ( KIND = rp_ ), DIMENSION( n ) :: D
   INTEGER ( KIND = ip_ ) :: j, status, S_ne
   D( 1 : n ) = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
! problem data complete
   CALL BSC_initialize( data, control, inform ) ! Initialize control parameters
   DO j = 1, 3
!  sparse co-ordinate storage format
     IF ( j == 1 ) THEN
       A_type = 'COORDINATE    '
       ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ) )
       A_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A_row = (/ 1, 1, 2, 2, 3, 3 /)
       A_col = (/ 1, 2, 3, 4, 1, 4 /)
! sparse row-wise storage format
     ELSE IF ( j == 2 ) THEN
       DEALLOCATE( A_row, A_col, A_val )
       A_type = 'SPARSE_BY_ROWS'
       ALLOCATE( A_val( a_ne ), A_col( a_ne ), A_ptr( m + 1 ) )
       A_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A_col = (/ 1, 2, 3, 4, 1, 4 /)
       A_ptr = (/ 1, 3, 5, 7 /)
! dense storage format
     ELSE IF ( j == 3 ) THEN
       DEALLOCATE( A_ptr, A_col, A_val )
       A_type = 'DENSE         '
       ALLOCATE( A_val( n * m ) )
       A_val = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,        &
                  1.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
     END IF

     CALL BSC_import( control, data, status, m, n,                             &
                      TRIM( A_type ), A_ne, A_row, A_col, A_ptr, S_ne )
     WRITE( 6, "( ' BSC_import exit status = ', I0 ) " ) status
     ALLOCATE( S_val( S_ne ), S_row( S_ne ), S_col( S_ne ) )
     CALL BSC_form_s( data, status, A_val, S_row, S_col, S_val )
     WRITE( 6, "( ' BSC_form_s exit status = ', I0 ) " ) status
     DEALLOCATE( S_val, S_row, S_col )

     CALL BSC_import( control, data, status, m, n,                             &
                      TRIM( A_type ), A_ne, A_row, A_col, A_ptr, S_ne )
     WRITE( 6, "( ' BSC_import exit status = ', I0 ) " ) status
     ALLOCATE( S_val( S_ne ), S_row( S_ne ), S_col( S_ne ), S_ptr( n + 1 ) )
     CALL BSC_form_s( data, status, A_val, S_row, S_col, S_val,                &
                      D = D, S_ptr = S_ptr )
     WRITE( 6, "( ' BSC_form_s exit status = ', I0 ) " ) status
     DEALLOCATE( S_val, S_row, S_col, S_ptr )
   END DO
   DEALLOCATE( A_val )

   CALL BSC_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_BSC_interface_test
