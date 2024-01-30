! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BSC_TEST
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BSC_precision
   IMPLICIT NONE
   TYPE ( BSC_data_type ) :: data
   TYPE ( BSC_control_type ) :: control
   TYPE ( BSC_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: m = 3, n = 4, a_ne = 6
   TYPE ( SMT_type ) :: A, S
   REAL ( KIND = rp_ ), DIMENSION( n ) :: D
   INTEGER ( KIND = ip_ ) :: i, j
   D( 1 : n ) = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
!  sparse co-ordinate storage format
   CALL SMT_put( A%type, 'COORDINATE', i )     ! storage for A
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   A%ne = a_ne
   A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)!Jacobian A
   A%row = (/ 1, 1, 2, 2, 3, 3 /)
   A%col = (/ 1, 2, 3, 4, 1, 4 /)
! problem data complete
   CALL BSC_initialize( data, control, inform ) ! Initialize control parameters

!  error exit tests

   WRITE( 6, "( ' error exit tests ', / ) " )
   CALL BSC_form( -1_ip_, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL BSC_form( m, 0_ip_, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL SMT_put( A%type, 'BOORDINATE', i )     ! storage for A
   CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL SMT_put( A%type, 'COORDINATE', i )     ! storage for A
   control%max_col = 1
   CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status

!  tests for normal entry

   control%max_col = - 1
   WRITE( 6, "( /, ' normal exit tests ', / ) " )

   DO j = 1, 3
!  sparse co-ordinate storage format
     IF ( j == 1 ) THEN
       DEALLOCATE( A%row, A%col, A%val )
       CALL SMT_put( A%type, 'COORDINATE', i )
       ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
       A%ne = a_ne
       A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A%row = (/ 1, 1, 2, 2, 3, 3 /)
       A%col = (/ 1, 2, 3, 4, 1, 4 /)
! sparse row-wise storage format
     ELSE IF ( j == 2 ) THEN
       DEALLOCATE( A%row, A%col, A%val )
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', i )
       ALLOCATE( A%val( a_ne ), A%col( a_ne ), A%ptr( m + 1 ) )
       A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       A%col = (/ 1, 2, 3, 4, 1, 4 /)
       A%ptr = (/ 1, 3, 5, 7 /)
! dense storage format
     ELSE IF ( j == 3 ) THEN
       DEALLOCATE( A%ptr, A%col, A%val )
       CALL SMT_put( A%type, 'DENSE', i )
       ALLOCATE( A%val( n * m ) )
       A%val = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,        &
                  1.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ /)
     END IF
     control%new_a = 2
     CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 1
     CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 2
     CALL BSC_form( m, n, A, S, data, control, inform, D = D ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 1
     CALL BSC_form( m, n, A, S, data, control, inform, D = D ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   END DO
   DEALLOCATE( A%val )

   CALL BSC_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_BSC_TEST
