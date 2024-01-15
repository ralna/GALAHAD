! THIS VERSION: GALAHAD 4.1 - 2023-02-25 AT 11:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_GSM_interface_test
   USE GALAHAD_USERDATA_precision
   USE GALAHAD_GSM_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( GSM_control_type ) :: control
   TYPE ( GSM_inform_type ) :: inform
   TYPE ( GSM_full_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER ( KIND = ip_ ) :: n, ne
   INTEGER ( KIND = ip_ ) :: i, s, data_storage_type, status, eval_status
   LOGICAL :: alive
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   REAL ( KIND = rp_ ) :: dum, f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, G, U, V
   CHARACTER ( len = 1 ) :: st

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

! start problem data

   n = 3 ; ne = 5 ! dimensions
   ALLOCATE( X( n ), G( n ) )
! problem data complete
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' basic test ', / )" )

   CALL GSM_initialize( data, control, inform )
!  control%print_level = 1
   X = 1.5_rp_  ! start from 1.5
   CALL GSM_import( control, data, status, n )
   CALL GSM_solve_with_mat( data, userdata, status, X, G, FUN, GRAD )
   CALL GSM_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( I6, ' iterations. Optimal objective value = ',    &
   &    F5.2, ' status = ', I0 )" ) inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A1, ': GSM_solve exit status = ', I0 ) " ) st, inform%status
   END IF
!  WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!  WRITE( 6, "( ' G ', 3ES12.5 )" ) G
   CALL GSM_terminate( data, control, inform )  ! delete internal workspace

   WRITE( 6, "( /, ' tests reverse-communication options ', / )" )

   f = 0.0_rp_
   ALLOCATE( U( n ), V( n ) ) ! reverse-communication input/output
   DO data_storage_type = 1, 5
     CALL GSM_initialize( data, control, inform )
!    control%print_level = 1
     X = 1.5_rp_  ! start from 1.5
     CALL GSM_import( control, data, status, n )
     DO ! reverse-communication loop
       CALL GSM_solve_reverse_with_mat( data, status, eval_status, X, f, G )
       SELECT CASE ( status )
       CASE ( 0 ) ! successful termination
         EXIT
       CASE ( : - 1 ) ! error exit
         EXIT
       CASE ( 2 ) ! evaluate f
         CALL FUN( eval_status, X, userdata, f )
       CASE ( 3 ) ! evaluate g
         CALL GRAD( eval_status, X, userdata, G )
       CASE DEFAULT
         WRITE( 6, "( ' the value ', I0, ' of status should not occur ')" )    &
           status
         EXIT
       END SELECT
     END DO
     CALL GSM_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A1, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A1, ': GSM_solve exit status = ', I0 ) " ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL GSM_terminate( data, control, inform )  ! delete internal workspace
   END DO

   DEALLOCATE( X, G )
   DEALLOCATE( userdata%real )

CONTAINS

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), INTENT( OUT ) :: f
   REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + COS( X( 1 ) )
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   G( 1 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) - SIN( X( 1 ) )
   G( 2 ) = 2.0_rp_ * ( X( 2 ) + X( 3 ) )
   G( 3 ) = 2.0_rp_ * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +               &
            2.0_rp_ * ( X( 2 ) + X( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE GRAD

   END PROGRAM GALAHAD_GSM_interface_test
