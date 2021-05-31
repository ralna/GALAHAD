! THIS VERSION: GALAHAD 2.6 - 26/06/2013 AT 08:30 GMT.
   PROGRAM GALAHAD_TRAL_test_deck
   USE GALAHAD_TRAL_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( TRAL_control_type ) :: control
   TYPE ( TRAL_inform_type ) :: inform
   TYPE ( TRAL_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   EXTERNAL :: FUN, GRAD, HESS, HESSPROD, PREC
   INTEGER :: i, s, scratch_out = 56
   logical :: alive
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
   REAL ( KIND = wp ) :: dum
! start problem data
   nlp%n = 1 ; nlp%H%ne = 1                     ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
             nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1 /) ; nlp%H%col = (/ 1 /)
! problem data complete   

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for s = - 1 ... - 40

   DO s = 1, 40

     IF ( s == - GALAHAD_error_allocate ) CYCLE
     IF ( s == - GALAHAD_error_deallocate ) CYCLE
!    IF ( s == - GALAHAD_error_restrictions ) CYCLE
     IF ( s == - GALAHAD_error_bad_bounds ) CYCLE
     IF ( s == - GALAHAD_error_primal_infeasible ) CYCLE
     IF ( s == - GALAHAD_error_dual_infeasible ) CYCLE
!    IF ( s == - GALAHAD_error_unbounded ) CYCLE
     IF ( s == - GALAHAD_error_no_center ) CYCLE
     IF ( s == - GALAHAD_error_analysis ) CYCLE
     IF ( s == - GALAHAD_error_factorization ) CYCLE
     IF ( s == - GALAHAD_error_solve ) CYCLE
     IF ( s == - GALAHAD_error_uls_analysis ) CYCLE
     IF ( s == - GALAHAD_error_uls_factorization ) CYCLE
     IF ( s == - GALAHAD_error_uls_solve ) CYCLE
!    IF ( s == - GALAHAD_error_preconditioner ) CYCLE
     IF ( s == - GALAHAD_error_ill_conditioned ) CYCLE
     IF ( s == - GALAHAD_error_tiny_step ) CYCLE
!    IF ( s == - GALAHAD_error_max_iterations ) CYCLE
!    IF ( s == - GALAHAD_error_cpu_limit ) CYCLE
     IF ( s == - GALAHAD_error_inertia ) CYCLE
     IF ( s == - GALAHAD_error_file ) CYCLE
     IF ( s == - GALAHAD_error_io ) CYCLE
     IF ( s == - GALAHAD_error_upper_entry ) CYCLE
     IF ( s == - GALAHAD_error_sort ) CYCLE
     IF ( s > 24 .AND. s < 40 ) CYCLE
     CALL TRAL_initialize( data, control, inform )  ! Initialize controls
!    control%print_level = 1
     inform%status = 1                           ! set for initial entry
     nlp%n = 1
     nlp%X = 1.0_wp                              ! start from one
     control%hessian_available = .FALSE.         ! Hessian prods will be used

     IF ( s == - GALAHAD_error_restrictions ) THEN
       nlp%n = 0
     ELSE IF ( s == - GALAHAD_error_preconditioner ) THEN
       control%preconditioner = - 3               ! User's preconditioner
     ELSE IF ( s == - GALAHAD_error_unbounded ) THEN
     ELSE IF ( s == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
     ELSE IF ( s == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0_wp
     END IF
     DO                                           ! Loop to solve problem
       CALL TRAL_solve( nlp, control, inform, data, userdata )
       SELECT CASE ( inform%status )              ! reverse communication
       CASE ( 2 )                                 ! Obtain the objective
         nlp%f = - nlp%X( 1 ) ** 2
         data%eval_status = 0                     ! record successful evaluation
         IF ( control%alive_unit > 0 .AND. s == 40 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( alive .AND. control%alive_unit > 0 ) THEN
             OPEN( control%alive_unit, FILE = control%alive_file,              &
                   FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
             REWIND control%alive_unit
             CLOSE( control%alive_unit, STATUS = 'DELETE' )
           END IF
         END IF
         IF ( s == - GALAHAD_error_cpu_limit ) THEN
           dum = 0.0_wp
           DO i = 1, 10000000
             dum = dum + 0.0000001_wp * i / ( i + 1 )
           END DO
           nlp%f = ( nlp%f + dum ) - dum
         END IF
       CASE ( 3 )                                 ! Obtain the gradient
         nlp%G( 1 ) = - 2.0_wp * nlp%X( 1 )
         data%eval_status = 0                     ! record successful evaluation
       CASE ( 5 )                                 ! Obtain Hessian-vector prod
         data%U( 1 ) = data%U( 1 ) - 2.0_wp * data%V( 1 )
         data%eval_status = 0                     ! record successful evaluation
       CASE ( 6 )                                 ! Apply the preconditioner
         data%U( 1 ) = - data%V( 1 )
       CASE DEFAULT                               ! Terminal exit from loop
         EXIT
       END SELECT
     END DO
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRAL_solve exit status = ', I6 ) " ) s, inform%status
     END IF

     CALL TRAL_terminate( data, control, inform )  ! delete internal workspace
   END DO

   control%subproblem_direct = .TRUE.         ! Use a direct method
   CALL TRAL_solve( nlp, control, inform, data, userdata,                      &
                   eval_F = FUN, eval_G = GRAD, eval_H = HESS )

   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col )

!  =========================
!  test of available options
!  =========================

! start problem data
   nlp%n = 3 ; nlp%H%ne = 5                  ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
             nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)           ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)           ! NB lower triangle
! problem data complete   
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' test of availible options ', / )" )

   DO i = 1, 7
     CALL TRAL_initialize( data, control, inform )     ! Initialize controls
!    control%print_level = 1
     inform%status = 1                        ! set for initial entry
     nlp%X = 1.0_wp                           ! start from one

     IF ( i == 1 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%trs_control%out = scratch_out
       control%trs_control%error = scratch_out
       control%trs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 2 ) THEN
       control%preconditioner = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%trs_control%out = scratch_out
       control%trs_control%error = scratch_out
       control%trs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 3 ) THEN
       control%preconditioner = 3
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
     ELSE IF ( i == 4 ) THEN
       control%preconditioner = 5
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
     ELSE IF ( i == 5 ) THEN
       control%preconditioner = - 2
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD,  eval_H = HESS )
     ELSE IF ( i == 6 ) THEN
       control%model = 1
       control%maxit = 1000
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD )
     ELSE IF ( i == 7 ) THEN
       control%model = 3
       control%maxit = 1000
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD )
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRAL_solve exit status = ', I6 ) " ) i, inform%status
     END IF

     CALL TRAL_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )

!  ============================
!  full test of generic problem
!  ============================

! start problem data
   nlp%n = 3 ; nlp%H%ne = 5                  ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                     &
                        nlp%H%col( nlp%H%ne ) )
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)            ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)            ! NB lower triangle
! problem data complete   
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   DO i = 1, 6
     CALL TRAL_initialize( data, control, inform )     ! Initialize controls
!    control%print_level = 1
     inform%status = 1                          ! set for initial entry
     nlp%X = 1.0_wp                             ! start from one

     IF ( i == 1 ) THEN
       control%subproblem_direct = .TRUE.       ! Use a direct method
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD, eval_H = HESS )
     ELSE IF ( i == 2 ) THEN
       control%hessian_available = .FALSE.      ! Hessian products will be used
       CALL TRAL_solve( nlp, control, inform, data, userdata,                  &
                       eval_F = FUN, eval_G = GRAD,  eval_HPROD = HESSPROD )
     ELSE IF ( i == 3 ) THEN
       control%hessian_available = .FALSE.      ! Hessian products will be used
       control%preconditioner = - 3             ! User's preconditioner
       CALL TRAL_solve( nlp, control, inform, data, userdata, eval_F = FUN,    &
              eval_G = GRAD, eval_HPROD = HESSPROD, eval_PREC = PREC )
     ELSE IF ( i == 4 .OR. i == 5 .OR. i == 6 ) THEN
       IF ( i == 4 ) THEN
         control%subproblem_direct = .TRUE.         ! Use a direct method
       ELSE
         control%hessian_available = .FALSE.        ! Hessian prods will be used
       END IF
       IF ( i == 6 ) control%preconditioner = - 3   ! User's preconditioner
       DO                                           ! Loop to solve problem
         CALL TRAL_solve( nlp, control, inform, data, userdata )
         SELECT CASE ( inform%status )              ! reverse communication
         CASE ( 2 )                                 ! Obtain the objective
           nlp%f = ( nlp%X( 1 ) + nlp%X( 3 ) + p ) ** 2 +                      &
                   ( nlp%X( 2 ) + nlp%X( 3 ) ) ** 2 + COS( nlp%X( 1 ) )
           data%eval_status = 0                   ! record successful evaluation
         CASE ( 3 )                               ! Obtain the gradient
           nlp%G( 1 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p ) -             &
                        SIN( nlp%X( 1 ) )
           nlp%G( 2 ) = 2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) )
           nlp%G( 3 ) = 2.0_wp * ( nlp%X( 1 ) + nlp%X( 3 ) + p ) +             &
                        2.0_wp * ( nlp%X( 2 ) + nlp%X( 3 ) )
           data%eval_status = 0                   ! record successful evaluation
         CASE ( 4 )                               ! Obtain the Hessian
           nlp%H%val( 1 ) = 2.0_wp - COS( nlp%X( 1 ) )
           nlp%H%val( 2 ) = 2.0_wp
           nlp%H%val( 3 ) = 2.0_wp
           nlp%H%val( 4 ) = 2.0_wp
           nlp%H%val( 5 ) = 4.0_wp
           data%eval_status = 0                  ! record successful evaluation
         CASE ( 5 )                              ! Obtain Hessian-vector prod
           data%U( 1 ) = data%U( 1 ) + 2.0_wp * ( data%V( 1 ) + data%V( 3 ) )  &
                           - COS( nlp%X( 1 ) ) * data%V( 1 )
           data%U( 2 ) = data%U( 2 ) + 2.0_wp * ( data%V( 2 ) + data%V( 3 ) )
           data%U( 3 ) = data%U( 3 ) + 2.0_wp * ( data%V( 1 ) + data%V( 2 ) +  &
                         2.0_wp * data%V( 3 ) )
           data%eval_status = 0                  ! record successful evaluation
         CASE ( 6 )                              ! Apply the preconditioner
           data%U( 1 ) = 0.5_wp * data%V( 1 )
           data%U( 2 ) = 0.5_wp * data%V( 2 )
           data%U( 3 ) = 0.25_wp * data%V( 3 )
           data%eval_status = 0                  ! record successful evaluation
         CASE DEFAULT                            ! Terminal exit from loop
           EXIT
         END SELECT
       END DO
     ELSE
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': TRAL_solve exit status = ', I6 ) " ) i, inform%status
     END IF

     CALL TRAL_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
   WRITE( 6, "( /, ' ***  This package needs much more work!!' )" )
   END PROGRAM GALAHAD_TRAL_test_deck

   SUBROUTINE FUN( status, X, userdata, f )     ! Objective function
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   f = ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) ** 2 +                         &
       ( X( 2 ) + X( 3 ) ) ** 2 + COS( X( 1 ) )
   status = 0
   RETURN
   END SUBROUTINE FUN

   SUBROUTINE GRAD( status, X, userdata, G )    ! gradient of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   G( 1 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) - SIN( X( 1 ) )
   G( 2 ) = 2.0_wp * ( X( 2 ) + X( 3 ) )
   G( 3 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + userdata%real( 1 ) ) +                &
            2.0_wp * ( X( 2 ) + X( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE HESS( status, X, userdata, Hval ) ! Hessian of the objective
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   Hval( 1 ) = 2.0_wp - COS( X( 1 ) )
   Hval( 2 ) = 2.0_wp
   Hval( 3 ) = 2.0_wp
   Hval( 4 ) = 2.0_wp
   Hval( 5 ) = 4.0_wp
   status = 0
   RETURN
   END SUBROUTINE HESS

   SUBROUTINE HESSPROD( status, X, userdata, U, V ) ! Hessian-vector product
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   U( 1 ) = U( 1 ) + 2.0_wp * ( V( 1 ) + V( 3 ) ) - COS( X( 1 ) ) * V( 1 )
   U( 2 ) = U( 2 ) + 2.0_wp * ( V( 2 ) + V( 3 ) )
   U( 3 ) = U( 3 ) + 2.0_wp * ( V( 1 ) + V( 2 ) + 2.0_wp * V( 3 ) )
   status = 0
   RETURN
   END SUBROUTINE HESSPROD

   SUBROUTINE PREC( status, X, userdata, U, V ) ! apply preconditioner
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   U( 1 ) = 0.5_wp * V( 1 )
   U( 2 ) = 0.5_wp * V( 2 )
   U( 3 ) = 0.25_wp * V( 3 )
   status = 0
   RETURN
   END SUBROUTINE PREC
