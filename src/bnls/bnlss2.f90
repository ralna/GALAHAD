   PROGRAM GALAHAD_BNLS_EXAMPLE2 !  GALAHAD 5.5 - 2026-05-04 AT 11:10 GMT.
   USE GALAHAD_BNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( BNLS_control_type ) :: control
   TYPE ( BNLS_inform_type ) :: inform
   TYPE ( BNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   TYPE ( REVERSE_type ) :: reverse
   INTEGER :: s
   INTEGER, PARAMETER :: n = 5, m_r = 4, jr_ne = 8
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp   ! parameter p
! start problem data
   nlp%n = n ; nlp%m_r = m_r ; nlp%Jr%ne = jr_ne  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%X_l( n ), nlp%X_u( n ) )
   nlp%X_l = 0.0_wp ; nlp%X_u = 1.0_wp          ! variables lie in [0,1]
!  nlp%X = [ 1.0_wp, 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp ]
   nlp%X = [ 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp ]
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )  ! specify co-ordinate storage
   ALLOCATE( nlp%Jr%val( jr_ne ), nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
   nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /)     ! Jacobian Jr(x)
   nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
! problem data complete ; solve using a Gauss-Newton model
   CALL BNLS_initialize( data, control, inform ) ! initialize control params
   control%jacobian_available = 2                ! jacobian is available
   control%print_level = 1
   control%print_obj = .TRUE.
   control%subproblem_solver = 1 ! use internal blls (2 for bllsb)
!  control%BLLS_control%print_level = 1
   control%BLLS_control%SBLS_control%definite_linear_solver = 'potr '
   control%BLLS_control%SBLS_control%symmetric_linear_solver = 'sytr '
!  control%BLLSB_control%print_level = 1
   control%BLLSB_control%symmetric_linear_solver = 'sytr '
   control%BLLSB_control%FDC_control%symmetric_linear_solver = 'sytr '
   inform%status = 1 ! set for initial entry
   DO
     CALL BNLS_solve( nlp, control, inform, data, userdata, reverse = reverse )
     SELECT CASE( inform%status )
     CASE ( 0 ) ! successful return
       WRITE( 6, "( ' BNLS: ', I0, ' iterations -',                            &
      &     ' optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
       inform%iter, inform%obj, nlp%X
       EXIT
     CASE( 2 ) ! evaluate residual
       nlp%R( 1 ) = nlp%X( 1 ) * nlp%X( 2 ) - p
       nlp%R( 2 ) = nlp%X( 2 ) * nlp%X( 3 ) - 1.0_wp
       nlp%R( 3 ) = nlp%X( 3 ) * nlp%X( 4 ) - 1.0_wp
       nlp%R( 4 ) = nlp%X( 4 ) * nlp%X( 5 ) - 1.0_wp
       reverse%eval_status = 0
     CASE( 3 ) ! evaluate Jacobian
       nlp%Jr%val( 1 ) = nlp%X( 2 )
       nlp%Jr%val( 2 ) = nlp%X( 1 )
       nlp%Jr%val( 3 ) = nlp%X( 3 )
       nlp%Jr%val( 4 ) = nlp%X( 2 )
       nlp%Jr%val( 5 ) = nlp%X( 4 )
       nlp%Jr%val( 6 ) = nlp%X( 3 )
       nlp%Jr%val( 7 ) = nlp%X( 5 )
       nlp%Jr%val( 8 ) = nlp%X( 4 )
       reverse%eval_status = 0
     CASE DEFAULT ! error returns
       WRITE( 6, "( ' BNLS_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END SELECT
   END DO
! delete internal workspace
   CALL BNLS_terminate( data, control, inform, reverse = reverse )
   DEALLOCATE( nlp%X_l, nlp%X_u, nlp%X, nlp%Z )
   DEALLOCATE( nlp%G, nlp%R, nlp%X_status )
   DEALLOCATE( nlp%Jr%type, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
   END PROGRAM GALAHAD_BNLS_EXAMPLE2
