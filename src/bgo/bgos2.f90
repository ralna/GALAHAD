   PROGRAM GALAHAD_BGO_EXAMPLE2  !  GALAHAD 4.0 - 2022-03-07 AT 13:30 GMT
   USE GALAHAD_BGO_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( BGO_control_type ) :: control
   TYPE ( BGO_inform_type ) :: inform
   TYPE ( BGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS, HESSPROD
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, h_ne = 5
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ), PARAMETER :: freq = 10.0_wp
   REAL ( KIND = wp ), PARAMETER :: mag = 1000.0_wp
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20    ! infinity
! start problem data
   nlp%pname = 'BGOSPEC'                        ! name
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   nlp%X = 0.0_wp                               ! start from zero
   nlp%X_l = -10.0_wp ; nlp%X_u = 0.5_wp ! search in [-10,1/2]
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H_row = (/ 1, 2, 3, 3, 3 /) ! Hessian H
   nlp%H_col = (/ 1, 2, 1, 2, 3 /) ! NB lower triangle
! problem data complete
   CALL BGO_initialize( data, control, inform ) ! Initialize control parameters
   control%TRB_control%subproblem_direct = .FALSE.  ! Use an iterative method
   control%attempts_max = 1000
   control%max_evals = 1000
   control%TRB_control%maxit = 10
! Solve the problem
   inform%status = 1                            ! set for initial entry
   DO ! Solve problem using reverse communication
     CALL BGO_solve( nlp, control, inform, data, userdata )
   
     SELECT CASE ( inform%status )
       CASE( 0 )  ! Successful return
         WRITE( 6, "( ' BGO: ', I0, ' evaluations -', /,                       &
        &     ' Best objective value found =', ES12.4, /,                      &
        &     ' Corresponding solution = ', ( 5ES12.4 ) )" )                   &
       inform%f_eval, inform%obj, nlp%X
       EXIT
     CASE ( 2 ) ! evaluate f
       nlp%f = ( nlp%X( 1 ) + nlp%X( 3 ) + p ) ** 2 +                          &
          ( nlp%X( 2 ) + nlp%X( 3 ) ) ** 2 + mag * COS( freq * nlp%X( 1 ) ) +  &
            nlp%X( 1 ) + nlp%X( 2 ) + nlp%X( 3 )
       data%eval_status = 0
     CASE ( 3 ) ! evaluate g
       nlp%G( 1 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p )                   &
                      - mag * freq * SIN( freq * nlp%X( 1 ) ) + 1.0_wp
       nlp%G( 2 ) = 2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) ) + 1.0_wp
       nlp%G( 3 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p )                   &
                      + 2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) ) + 1.0_wp
       data%eval_status = 0
     CASE ( 4 ) ! evaluate H
       nlp%H%val( 1 ) = 2.0_wp - mag * freq * freq * COS( freq * nlp%X( 1 ) )
       nlp%H%val( 2 ) = 2.0_wp
       nlp%H%val( 3 ) = 2.0_wp
       nlp%H%val( 4 ) = 2.0_wp
       nlp%H%val( 5 ) = 4.0_wp
       data%eval_status = 0
     CASE ( 5 ) ! evaluate the product u = Hv
       data%U( 1 ) = data%U( 1 ) + ( 2.0_wp - mag * freq * freq *              &
              COS( freq * nlp%X( 1 ) ) ) * data%V( 1 ) + 2.0_wp * data%V( 3 )
       data%U( 2 ) = data%U( 2 ) + 2.0_wp * ( data%V( 2 ) + data%V( 3 ) )
       data%U( 3 ) = data%U( 3 ) + 2.0_wp * ( data%V( 1 ) + data%V( 2 )        &
                       + 2.0_wp * data%V( 3 ) )
       data%eval_status = 0
     CASE DEFAULT ! Error returns
       WRITE( 6, "( ' BGO_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END SELECT
   END DO
   CALL BGO_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col )
   END PROGRAM GALAHAD_BGO_EXAMPLE2

