! THIS VERSION: GALAHAD 5.3 - 2025-07-19 AT 13:10 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_EXPO_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_EXPO_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( EXPO_control_type ) :: control
   TYPE ( EXPO_inform_type ) :: inform
   TYPE ( EXPO_full_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER ( KIND = ip_ ) :: n, m, J_ne, H_ne
   INTEGER ( KIND = ip_ ) :: data_storage_type, status
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20    ! infinity
   REAL ( KIND = rp_ ), PARAMETER :: p = 9.0_rp_
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u, Z, GL
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: J_row, J_col, J_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: J_val, J_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_val, H_dense, H_diag
   CHARACTER ( len = 2 ) :: st

! set up problem data

   n = 2 ;  m = 5 ; J_ne = 10 ; H_ne = 2
   ALLOCATE( X( n ), X_l( n ), X_u( n ), Z( n ), GL( n ), H_diag( n ) )
   ALLOCATE( C( m ), C_l( m ), C_u( m ), Y( m ) )
   X_l = - 50.0_rp_ ; X_u = 50.0_rp_     ! variable bounds
   C_l = 0.0_rp_ ; C_u = infinity        ! constraint bounds
   ALLOCATE( J_val( J_ne ), J_row( J_ne ), J_col( J_ne ), J_ptr( m + 1 ) )
   J_row = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 /)
   J_col = (/ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 /)
   J_ptr = (/ 1, 3, 5, 7, 9, 11 /)
   ALLOCATE( H_val( H_ne ), H_row( H_ne ), H_col( H_ne ), H_ptr( n + 1 ) )
   H_row = (/ 1, 2 /)
   H_col = (/ 1, 2 /) 
   H_ptr = (/ 1, 2, 3 /)
   ALLOCATE( J_dense( m * n ), H_dense( n * ( n + 1 ) / 2 ) )

   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of storage formats', / )" )

   DO data_storage_type = 1, 3
!  DO data_storage_type = 1, 1
     CALL EXPO_initialize( data, control, inform )
     control%max_it = 20
     control%max_eval = 100
!    control%print_level = 1
!    control%tru_control%print_level = 1
     control%stop_abs_p = 1.0D-5
     control%stop_abs_d = 1.0D-5
     control%stop_abs_c = 1.0D-5
     CALL WHICH_sls( control )
     X( 1 ) = 3.0_rp_ ; X( 2 ) = 1.0_rp_
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' C'
       CALL EXPO_import( control, data, status, n, m,                          &
                         'coordinate', J_ne, J_row, J_col, null,               &
                         'coordinate', H_ne, H_row, H_col, null )
       CALL EXPO_solve_hessian_direct( data, userdata, status,                 &
                                       C_l, C_u, X_l, X_u, X, Y, Z, C, GL,     &
                                       eval_FC = FC, eval_GJ = GJ,             &
                                       eval_HL = HL )
     CASE ( 2 ) ! sparse by rows
       st = ' R'
       CALL EXPO_import( control, data, status, n, m,                          &
                         'sparse_by_rows', J_ne, null, J_col, J_ptr,           &
                         'sparse_by_rows', H_ne, null, H_col, H_ptr )
       CALL EXPO_solve_hessian_direct( data, userdata, status,                 &
                                       C_l, C_u, X_l, X_u, X, Y, Z, C, GL,     &
                                       eval_FC = FC, eval_GJ = GJ,             &
                                       eval_HL = HL )
     CASE ( 3 ) ! dense
       st = ' D'
       CALL EXPO_import( control, data, status, n, m,                          &
                         'dense', J_ne, null, null, null,                      &
                         'dense', H_ne, null, null, null )
       CALL EXPO_solve_hessian_direct( data, userdata, status,                 &
                                       C_l, C_u, X_l, X_u, X, Y, Z, C, GL,     &
                                       eval_FC = FC, eval_GJ = GJ_dense,       &
                                       eval_HL = HL_dense )
     END SELECT
     CALL EXPO_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F5.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A2, ': EXPO_solve exit status = ', I0 )" ) st, inform%status
     END IF
!    WRITE( 6, "( ' X ', 3ES12.5 )" ) X
!    WRITE( 6, "( ' G ', 3ES12.5 )" ) G
     CALL EXPO_terminate( data, control, inform )  ! delete internal workspace
   END DO

   DEALLOCATE( X, Y, Z, C, GL )
   DEALLOCATE( J_val, J_row, J_col, J_ptr, J_dense )
   DEALLOCATE( H_val, H_row, H_col, H_ptr, H_dense, H_diag, userdata%real )

CONTAINS

     SUBROUTINE WHICH_sls( control )
     TYPE ( EXPO_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%SSLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%TRU_control%TRS_control%definite_linear_solver                    &
       = definite_linear_solver
     control%TRU_control%TRS_control%symmetric_linear_solver                   &
       = symmetric_linear_solver
     END SUBROUTINE WHICH_sls

     SUBROUTINE FC( status, X, userdata, F, C )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( kind = rp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( kind = rp ), OPTIONAL, INTENT( OUT ) :: F
     REAL ( kind = rp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( kind = rp ) :: r
     r = userdata%real( 1 )
     f = X( 1 ) ** 2 + X( 2 ) ** 2
     C( 1 ) = X( 1 ) + X( 2 ) - 1.0_rp
     C( 2 ) = X( 1 ) ** 2 + X( 2 ) ** 2 - 1.0_rp
     C( 3 ) = r * X( 1 ) ** 2 + X( 2 ) ** 2 - r
     C( 4 ) = X( 1 ) ** 2 - X( 2 )
     C( 5 ) = X( 2 ) ** 2 - X( 1 )
     status = 0
     END SUBROUTINE FC

     SUBROUTINE GJ( status, X, userdata, G, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
     REAL ( KIND = rp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( kind = rp ) :: r
     r = userdata%real( 1 )
     G( 1 ) = 2.0_rp * X( 1 )
     G( 2 ) = 2.0_rp * X( 2 )
     J_val( 1 ) = 1.0_rp
     J_val( 2 ) = 1.0_rp
     J_val( 3 ) = 2.0_rp * X( 1 )
     J_val( 4 ) = 2.0_rp * X( 2 )
     J_val( 5 ) = 2.0_rp * r * X( 1 )
     J_val( 6 ) = 2.0_rp * X( 2 )
     J_val( 7 ) = 2.0_rp * X( 1 )
     J_val( 8 ) = - 1.0_rp
     J_val( 9 ) = - 1.0_rp
     J_val( 10 ) = 2.0_rp * X( 2 )
     END SUBROUTINE GJ

     SUBROUTINE HL( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( kind = rp ) :: r
     r = userdata%real( 1 )
     H_val( 1 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + r * Y( 3 ) +  Y( 4 ) )
     H_val( 2 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + Y( 3 ) + Y( 5 ) )
     END SUBROUTINE HL

     SUBROUTINE GJ_dense( status, X, userdata, G, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
     REAL ( KIND = rp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( kind = rp ) :: r
     r = userdata%real( 1 )
     G( 1 ) = 2.0_rp * X( 1 )
     G( 2 ) = 2.0_rp * X( 2 )
     J_val( 1 ) = 1.0_rp
     J_val( 2 ) = 1.0_rp
     J_val( 3 ) = 2.0_rp * X( 1 )
     J_val( 4 ) = 2.0_rp * X( 2 )
     J_val( 5 ) = 2.0_rp * r * X( 1 )
     J_val( 6 ) = 2.0_rp * X( 2 )
     J_val( 7 ) = 2.0_rp * X( 1 )
     J_val( 8 ) = - 1.0_rp
     J_val( 9 ) = - 1.0_rp
     J_val( 10 ) = 2.0_rp * X( 2 )
     END SUBROUTINE GJ_dense

     SUBROUTINE HL_dense( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( kind = rp ) :: r
     r = userdata%real( 1 )
     H_val( 1 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + r * Y( 3 ) +  Y( 4 ) )
     H_val( 2 ) = 0.0_rp
     H_val( 3 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + Y( 3 ) + Y( 5 ) )
     END SUBROUTINE HL_dense

   END PROGRAM GALAHAD_EXPO_interface_test
