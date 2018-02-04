   PROGRAM GALAHAD_TRAL3_EXAMPLE  !  GALAHAD 2.6 - 26/06/2013 AT 08:15 GMT
   USE GALAHAD_TRAL_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( TRAL_control_type ) :: control
   TYPE ( TRAL_inform_type ) :: inform
   TYPE ( TRAL_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, h_ne = 5
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
! start problem data
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ) )
   nlp%X = 1.0_wp                               ! start from one
! problem data complete   
   CALL TRAL_initialize( data, control, inform ) ! Initialize control parameters
   control%hessian_available = .FALSE.          ! Hessian products will be used
!  control%psls_control%preconditioner = - 3    ! Apply uesr's preconditioner
   inform%status = 1                            ! Set for initial entry
   DO                                           ! Loop to solve problem
     CALL TRAL_solve( nlp, control, inform, data, userdata )
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
     WRITE( 6, "( ' TRAL: ', I0, ' iterations -',                               &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                         ! Error returns
     WRITE( 6, "( ' TRAL_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL TRAL_terminate( data, control, inform )  ! Delete internal workspace
   END PROGRAM GALAHAD_TRAL3_EXAMPLE
