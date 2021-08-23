   PROGRAM GALAHAD_TRB3_EXAMPLE  !  GALAHAD 3.3 - 29/07/2021 AT 07:45 GMT
   USE GALAHAD_TRB_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( TRB_control_type ) :: control
   TYPE ( TRB_inform_type ) :: inform
   TYPE ( TRB_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, h_ne = 5
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20    ! infinity
! start problem data
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   nlp%X = 1.0_wp                               ! start from one
   nlp%X_l( : n )  = (/ - infinity, - infinity, 0.0_wp /) ; nlp%X_u = 1.1_wp
! sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)              ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)              ! NB lower triangle
! problem data complete   
   CALL TRB_initialize( data, control, inform ) ! Initialize control parameters
!  control%hessian_available = .FALSE.          ! Hessian products will be used
!  control%psls_control%preconditioner = - 3    ! Apply uesr's preconditioner
   inform%status = 1                            ! Set for initial entry
   DO                                           ! Loop to solve problem
     CALL TRB_solve( nlp, control, inform, data, userdata )
     SELECT CASE ( inform%status )              ! reverse communication
     CASE ( 2 )                                 ! Obtain the objective function
       nlp%f = ( nlp%X( 1 ) + nlp%X( 3 ) + p ) ** 2 +                          &
               ( nlp%X( 2 ) + nlp%X( 3 ) ) ** 2 + COS( nlp%X( 1 ) )
       data%eval_status = 0                     ! record successful evaluation
     CASE ( 3 )                                 ! Obtain the gradient
       nlp%G( 1 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p ) - SIN( nlp%X( 1 ) )
       nlp%G( 2 ) = 2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) )
       nlp%G( 3 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p ) +                 &
                    2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) )
       data%eval_status = 0                     ! record successful evaluation
     CASE ( 4 )                                 ! Obtain Hessian evaluation
        nlp%H%val( 1 ) = 2.0_wp - COS( nlp%X( 1 ) )
        nlp%H%val( 2 ) = 2.0_wp
        nlp%H%val( 3 ) = 2.0_wp
        nlp%H%val( 4 ) = 2.0_wp
        nlp%H%val( 5 ) = 4.0_wp
       data%eval_status = 0                     ! record successful evaluation
     CASE ( 5 )                                 ! Obtain Hessian-vector product
       data%U( 1 ) = data%U( 1 ) + 2.0_wp * ( data%V( 1 ) + data%V( 3 ) ) -    &
                     COS( nlp%X( 1 ) ) * data%V( 1 )
       data%U( 2 ) = data%U( 2 ) + 2.0_wp * ( data%V( 2 ) + data%V( 3 ) )
       data%U( 3 ) = data%U( 3 ) + 2.0_wp * ( data%V( 1 ) + data%V( 2 ) +      &
                     2.0_wp * data%V( 3 ) )
       data%eval_status = 0                     ! record successful evaluation
     CASE ( 6 )                                 ! Apply the preconditioner
       data%U( 1 ) = 0.5_wp * data%V( 1 )
       data%U( 2 ) = 0.5_wp * data%V( 2 )
       data%U( 3 ) = 0.25_wp * data%V( 3 )
       data%eval_status = 0                     ! record successful evaluation
     CASE DEFAULT                               ! Terminal exit from loop
       EXIT
     END SELECT
   END DO
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' TRB: ', I0, ' iterations -',                               &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' TRB_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL TRB_terminate( data, control, inform )  ! Delete internal workspace
   END PROGRAM GALAHAD_TRB3_EXAMPLE
