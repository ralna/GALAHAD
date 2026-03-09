   PROGRAM GALAHAD_SNLS_EXAMPLE2 !  GALAHAD 5.5 - 2026-03-07 AT 15:00 GMT
   USE GALAHAD_SNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   TYPE ( REVERSE_type ) :: reverse
   INTEGER :: s
   INTEGER, PARAMETER :: n = 5, m_r = 4, m_c = 2, jr_ne = 8
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp   ! parameter p
! start problem data
   nlp%n = n ; nlp%m_r = m_r ; nlp%m_c = m_c ; nlp%Jr%ne = jr_ne  ! dimensions
   ALLOCATE( nlp%COHORT( n ), nlp%X( n ) )
   nlp%COHORT = [ 1, 2, 0, 1, 2 ]
!  nlp%X = [ 1.0_wp, 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp ]
   nlp%X = [ 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp ]
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )  ! specify co-ordinate storage
   ALLOCATE( nlp%Jr%val( jr_ne ), nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
   nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /)     ! Jacobian Jr(x)
   nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
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
   inform%status = 1 ! set for initial entry
   DO
     CALL SNLS_solve( nlp, control, inform, data, userdata, reverse = reverse )
     SELECT CASE( inform%status )
     CASE ( 0 ) ! successful return
       WRITE( 6, "( ' SNLS: ', I0, ' iterations -',                            &
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
       WRITE( 6, "( ' SNLS_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END SELECT
   END DO
   CALL SNLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%R, nlp%COHORT )
   DEALLOCATE( nlp%Jr%type, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
   END PROGRAM GALAHAD_SNLS_EXAMPLE2
