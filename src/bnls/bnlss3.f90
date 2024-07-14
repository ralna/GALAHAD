   PROGRAM GALAHAD_BNLS_EXAMPLE3 !  GALAHAD 5.1 - 2024-07-14 AT 14:00 GMT
   USE GALAHAD_BNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( BNLS_control_type ) :: control
   TYPE ( BNLS_inform_type ) :: inform
   TYPE ( BNLS_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: s
   INTEGER, PARAMETER :: m = 2, n = 3, j_ne = 4, h_ne = 3, p_ne = 3
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp  ! parameter p
! start problem data
   nlp%n = n ; nlp%m = m ; nlp%J%ne = j_ne ; nlp%H%ne = h_ne   ! dimensions
   ALLOCATE( nlp%X( n ), nlp%X_l( n ), nlp%X_u( n ), nlp%C( m ) )
   nlp%X = (/ 1.0_wp, 1.0_wp, 1.0_wp /)         ! start from (-1,1,1)
   nlp%X_l = 0.0_wp ; nlp%X_u = 1.0_wp          ! variables lie in [0,1]
! problem data complete ; solve using a Newton model
   CALL BNLS_initialize( data, control, inform ) ! Initialize control params
   control%jacobian_available = 1               ! only Jacobian-vector products
   control%hessian_available = 1                ! only Hessian-vector products
   control%model = 4                            ! use the Newton model
   inform%status = 1                            ! set for initial entry
10 CONTINUE
   CALL BNLS_solve( nlp, control, inform, data, userdata )
   SELECT CASE ( inform%status ) !  is more information required?
   CASE ( 0 ) !  successful call
     WRITE( 6, "( ' BNLS: ', I0, ' iterations -',                              &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   CASE ( : - 1 ) !  unsuccessful call
     WRITE( 6, "( ' BNLS_solve exit status = ', I6 ) " ) inform%status
   CASE ( 2 )
     nlp%C( 1 ) = nlp%X( 3 ) * nlp%X( 1 ) ** 2 + P
     nlp%C( 2 ) = nlp%X( 2 ) ** 2 + nlp%X( 3 )
     data%eval_status = 0
     GO TO 10  !  return to BNLS_solve
   CASE ( 5 )
     IF ( data%transpose ) THEN
       data%U( 1 ) = data%U( 1 ) + 2.0_wp * nlp%X( 1 ) * nlp%X( 3 ) *          &
         data%V( 1 )
       data%U( 2 ) = data%U( 2 ) + 2.0_wp * nlp%X( 2 ) * data%V( 2 )
       data%U( 3 ) = data%U( 3 ) + data%V( 1 ) * nlp%X( 1 ) ** 2 + data%V( 2 )
     ELSE
       data%U( 1 ) = data%U( 1 ) + 2.0_wp * nlp%X( 1 ) * nlp%X( 3 ) *          &
         data%V( 1 ) + data%V( 3 ) * nlp%X( 1 ) ** 2
       data%U( 2 ) = data%U( 2 ) + 2.0_wp * nlp%X( 2 ) * data%V( 2 )           &
                                 + data%V( 3 )
     END IF
     data%eval_status = 0
     GO TO 10  !  return to BNLS_solve
   CASE ( 6 )
     data%U( 1 ) = data%U( 1 ) + 2.0_wp * data%Y( 1 ) *                        &
       ( nlp%X( 3 ) * data%V( 1 ) + nlp%X( 1 ) * data%V( 3 ) )
     data%U( 2 ) = data%U( 2 ) + 2.0_wp * data%Y( 2 ) * data%V( 2 )
     data%U( 3 ) = data%U( 3 ) + 2.0_wp * data%Y( 1 ) *                        &
        nlp%X( 1 ) * data%V( 1 )
     data%eval_status = 0
     GO TO 10  !  return to BNLS_solve
   END SELECT
   CALL BNLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G nlp%X_l, nlp%X_u )
   END PROGRAM GALAHAD_BNLS_EXAMPLE3
