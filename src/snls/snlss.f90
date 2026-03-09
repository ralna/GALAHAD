   PROGRAM GALAHAD_SNLS_EXAMPLE !  GALAHAD 5.5 - 2026-03-07 AT 15:00 GMT
   USE GALAHAD_SNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   EXTERNAL :: EVALR, EVALJr
   INTEGER :: s
   INTEGER, PARAMETER :: n = 5, m_r = 4, m = 2, jr_ne = 8
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp   ! parameter p
! start problem data
   nlp%n = n ; nlp%m_r = m_r ; nlp%m_c = m ; nlp%Jr%ne = jr_ne  ! dimensions
   ALLOCATE( nlp%COHORT( n ), nlp%X( n ) )
   nlp%COHORT = [ 1, 2, 0, 1, 2 ]
   nlp%X = [ 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp ]
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )  ! specify co-ordinate storage
   ALLOCATE( nlp%Jr%val( jr_ne ), nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
   nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /)     ! Jacobian Jr(x)
   nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
   ALLOCATE( userdata%real( 1 ) )                ! allocate space for parameter
   userdata%real( 1 ) = p                        ! record parameter, p
! problem data complete ; solve using a Gauss-Newton model
   CALL SNLS_initialize( data, control, inform ) ! initialize control params
   control%jacobian_available = 2                ! jacobian is available
   control%print_level = 1
   control%print_obj = .TRUE.
   control%subproblem_solver = 1 ! use internal slls (2 for sllsb)
!  control%SLLS_control%print_level = 1
   control%SLLS_control%SBLS_control%definite_linear_solver = 'potr '
   control%SLLS_control%SBLS_control%symmetric_linear_solver = 'sytr '
!  control%SLLSB_control%print_level = 1
   control%SLLSB_control%symmetric_linear_solver = 'sytr '
   control%SLLSB_control%FDC_control%symmetric_linear_solver = 'sytr '
   inform%status = 1                             ! set for initial entry
   CALL SNLS_solve( nlp, control, inform, data, userdata,                      &
                    eval_R = EVALR, eval_Jr = EVALJr )
   IF ( inform%status == 0 ) THEN                ! successful return
     WRITE( 6, "( ' SNLS: ', I0, ' iterations -',                              &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                          ! Error returns
     WRITE( 6, "( ' SNLS_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL SNLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%R, nlp%COHORT, userdata%real )
   DEALLOCATE( nlp%Jr%type, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
   END PROGRAM GALAHAD_SNLS_EXAMPLE

   SUBROUTINE EVALR( status, X, userdata, R ) ! residual
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: R
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: p
   p = userdata%real( 1 )
   R( 1 ) = X( 1 ) * X( 2 ) - p
   R( 2 ) = X( 2 ) * X( 3 ) - 1.0_wp
   R( 3 ) = X( 3 ) * X( 4 ) - 1.0_wp
   R( 4 ) = X( 4 ) * X( 5 ) - 1.0_wp
   status = 0
   RETURN
   END SUBROUTINE EVALR

   SUBROUTINE EVALJr( status, X, userdata, Jr_val ) ! Jacobian
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: p
   p = userdata%real( 1 )
   Jr_val( 1 ) = X( 2 )
   Jr_val( 2 ) = X( 1 )
   Jr_val( 3 ) = X( 3 )
   Jr_val( 4 ) = X( 2 )
   Jr_val( 5 ) = X( 4 )
   Jr_val( 6 ) = X( 3 )
   Jr_val( 7 ) = X( 5 )
   Jr_val( 8 ) = X( 4 )
   status = 0
   RETURN
   END SUBROUTINE EVALJr

