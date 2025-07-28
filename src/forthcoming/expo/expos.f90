   PROGRAM GALAHAD_EXPO_EXAMPLE  !  GALAHAD 5.3 - 2025-07-25 AT 11:15 GMT.
   USE GALAHAD_EXPO_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: rp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( EXPO_control_type ) :: control
   TYPE ( EXPO_inform_type ) :: inform
   TYPE ( EXPO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FC, GJ, HL
   INTEGER :: s
   INTEGER, PARAMETER :: n = 2, m = 5, j_ne = 10, h_ne = 2
   REAL ( KIND = rp ), PARAMETER :: p = 9.0_rp
   REAL ( KIND = rp ), PARAMETER :: infinity = 10.0_rp ** 20    ! infinity
! start problem data
   nlp%pname = 'HS23'                           ! name
   nlp%n = n ; nlp%m = m ; nlp%H%ne = h_ne      ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   ALLOCATE( nlp%C( m ), nlp%C_l( m ), nlp%C_u( m ) )
   nlp%X( 1 ) = 3.0_rp ; nlp%X( 2 ) = 1.0_rp
   nlp%X_l = - 50.0_rp ; nlp%X_u = 50.0_rp      ! variable bounds
   nlp%C_l = 0.0_rp ; nlp%C_u = infinity        ! constraint bounds
!  sparse row-wise storage format for the Jacobian
   CALL SMT_put( nlp%J%type, 'SPARSE_BY_ROWS', s ) ! specify sparse row storage
   ALLOCATE( nlp%J%val( j_ne ), nlp%J%col( j_ne ), nlp%H%ptr( m + 1 ) )
   nlp%J%col = (/ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 /)  ! Jacobian J
   nlp%J%ptr = (/ 1, 3, 5, 7, 9, 11 /)
!  sparse co-ordinate storage format for the Hessian
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 2 /)              ! Hessian H
   nlp%H%col = (/ 1, 2 /)              ! NB lower triangle
! problem data complete
   ALLOCATE( userdata%real( 1 ) )                ! allocate space for parameter
   userdata%real( 1 ) = p                        ! record parameter, p
   CALL EXPO_initialize( data, control, inform ) ! initialize control parameters
   control%subproblem_direct = .TRUE.
   control%max_it = 20
   control%max_eval = 100
!  control%print_level = 1
!  control%tru_control%print_level = 1
   control%stop_abs_p = 1.0D-5
   control%stop_abs_d = 1.0D-5
   control%stop_abs_c = 1.0D-5
   inform%status = 1                             ! set for initial entry
   CALL EXPO_solve( nlp, control, inform, data, userdata, eval_FC = FC,        &
                    eval_GJ = GJ, eval_HL = HL ) ! solve problem
   IF ( inform%status == 0 ) THEN                ! successful return
     WRITE( 6, "( ' EXPO: ', I0, ' major iterations -',                        &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                          ! error returns
     WRITE( 6, "( ' EXPO_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL EXPO_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%GL, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%col, nlp%J%ptr )
   DEALLOCATE( nlp%C, nlp%X_l, nlp%X_u, nlp%C_l, nlp%C_u, nlp%G )
   END PROGRAM GALAHAD_EXPO_EXAMPLE

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
   H_val( 1 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + r * Y( 3 ) + Y( 4 ) )
   H_val( 2 ) = 2.0_rp - 2.0_rp * ( Y( 2 ) + Y( 3 ) + Y( 5 ) )
   END SUBROUTINE HL
