! THIS VERSION: GALAHAD 3.3 - 05/05/2021 AT 14:15 GMT
   PROGRAM GALAHAD_NLS_test_deck
   USE GALAHAD_NLS_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( NLS_control_type ) :: control
   TYPE ( NLS_inform_type ) :: inform
   TYPE ( NLS_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
   REAL ( KIND = wp ), PARAMETER :: p = 1.0_wp
   REAL ( KIND = wp ), PARAMETER :: mult = 1.0_wp
!  EXTERNAL :: RES, JAC, HESS, JACPROD, HESSPROD, RHESSPRODS, SCALE
   INTEGER :: i, j, store, s, model, scaling, rev, usew
   CHARACTER ( len = 1 ) :: storage
   INTEGER :: scratch_out = 56
   logical :: alive
   REAL ( KIND = wp ) :: dum

!  GO TO 10

! start problem data

   nlp%n = 1 ;  nlp%m = 1 ; nlp%J%ne = 1 ; nlp%H%ne = 1 ; nlp%P%ne = 1
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ), W( nlp%m ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne))
   nlp%J%row = (/ 1 /)              ! Jacobian J
   nlp%J%col = (/ 1 /)
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne))
   nlp%H%row = (/ 1 /)                       ! Hessian H
   nlp%H%col = (/ 1 /)                       ! NB lower triangle only
   ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ), nlp%P%ptr( nlp%m+1))
   nlp%P%row = (/ 1 /)                       ! Hessian products
   nlp%P%ptr = (/ 1, 2 /)
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

! problem data complete

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

!  tests for s = - 1 ... - 40

   DO i = 1, 18
     CALL NLS_initialize( data, control, inform ) ! Initialize control params
!    control%print_level = 4
!    control%RQS_control%print_level = 4
!    control%GLRT_control%print_level = 4

!  choose error test

     IF ( i == 1 ) THEN
       nlp%n = 0 ; nlp%m = 0
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 2 ) THEN
       nlp%n = 1 ; nlp%m = 1
       control%jacobian_available = - 1
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 3 ) THEN
       W( 1 ) = - 1.0_wp
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 4 ) THEN
       IF ( ALLOCATED( nlp%J%type ) ) DEALLOCATE( nlp%J%type )
       CALL SMT_put( nlp%J%type, 'WRONG', s )
       control%model = 3
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 5 ) THEN
       s = GALAHAD_error_restrictions
       control%model = 6
     ELSE IF ( i == 6 ) THEN
       IF ( ALLOCATED( nlp%J%type ) ) DEALLOCATE( nlp%J%type )
       CALL SMT_put( nlp%J%type, 'COORDINATE', s )
       IF ( ALLOCATED( nlp%H%type ) ) DEALLOCATE( nlp%H%type )
       CALL SMT_put( nlp%H%type, 'WRONG', s )
       control%model = 4
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 7 ) THEN
       control%model = 7
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 8 ) THEN
       IF ( ALLOCATED( nlp%H%type ) ) DEALLOCATE( nlp%H%type )
       CALL SMT_put( nlp%H%type, 'COORDINATE', s )
       IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )
       CALL SMT_put( nlp%P%type, 'WRONG', s )
       control%model = 6
       s = GALAHAD_error_restrictions
     ELSE IF ( i == 9 ) THEN
       IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )
       control%stop_c_absolute = 0.0_wp
       control%stop_c_relative = 0.0_wp
       control%stop_g_absolute = 0.0_wp
       control%stop_g_relative = 0.0_wp
       s = GALAHAD_error_tiny_step
     ELSE IF ( i == 10 ) THEN
       control%stop_c_absolute = 0.0_wp
       control%stop_c_relative = 0.0_wp
       control%stop_g_absolute = 0.0_wp
       control%stop_g_relative = 0.0_wp
       control%model = 6
       s = GALAHAD_error_tiny_step
     ELSE IF ( i == 11 ) THEN
       control%maxit = 0
       s = GALAHAD_error_max_iterations
     ELSE IF ( i == 12 ) THEN
       control%maxit = 0
       control%model = 6
       s = GALAHAD_error_max_iterations
     ELSE IF ( i == 13 ) THEN
       control%cpu_time_limit = 0.0_wp
       s = GALAHAD_error_cpu_limit
     ELSE IF ( i == 14 ) THEN
       control%cpu_time_limit = 0.0_wp
       control%model = 6
       s = GALAHAD_error_cpu_limit
     ELSE IF ( i == 15 ) THEN
       CYCLE ! NAG doesn't cope
       s = GALAHAD_error_evaluation
     ELSE IF ( i == 16 ) THEN
       control%model = 6
       s = GALAHAD_error_evaluation
       CYCLE ! NAG doesn't cope
     ELSE IF ( i == 17 ) THEN
       s = GALAHAD_error_alive
     ELSE IF ( i == 18 ) THEN
       control%model = 6
       s = GALAHAD_error_alive
     END IF
     IF ( i >= 4 ) control%jacobian_available = 2
     IF ( i >= 6 ) control%hessian_available = 2

     inform%status = 1                            ! set for initial entry
     nlp%n = 1
     nlp%X = 1.0_wp                               ! start from one
     control%out = scratch_out
     control%error = scratch_out
     OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )

     DO                                           ! Loop to solve problem
!      write(6,*) ' problem ', s, 'in staus', inform%status
       SELECT CASE( i )
       CASE( 3 )
         CALL NLS_solve( nlp, control, inform, data, userdata, W = W )
       CASE DEFAULT
         CALL NLS_solve( nlp, control, inform, data, userdata )
       END SELECT
!      write(6,*) 'out status ', inform%status
       SELECT CASE ( inform%status )              ! reverse communication
       CASE ( 2 )                      ! Obtain the residuals
         nlp%C( 1 ) = mult * nlp%X( 1 )
         data%eval_status = 0 ! record successful evaluation
         SELECT CASE( s )     !  try to force error conditions
         CASE ( GALAHAD_error_cpu_limit ) ! try to raise cpu limit
           dum = 0.0_wp
           DO j = 1, 10000000
             dum = dum + 0.0000001_wp * j / ( j + 1 )
           END DO
           nlp%C( 1 ) = ( nlp%C( 1 ) + dum ) - dum
         CASE( GALAHAD_error_evaluation ) !  set a NaN
            nlp%C( 1 ) = 2.0_wp ** 128
!           nlp%C( 1 ) = 0.0_wp
!           nlp%C( 1 ) = nlp%C( 1 ) / nlp%C( 1 )
         CASE ( GALAHAD_error_alive ) ! remove alive file
           IF ( control%alive_unit > 0 ) THEN
             INQUIRE( FILE = control%alive_file, EXIST = alive )
             IF ( alive .AND. control%alive_unit > 0 ) THEN
               OPEN( control%alive_unit, FILE = control%alive_file,            &
                     FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
               REWIND control%alive_unit
               CLOSE( control%alive_unit, STATUS = 'DELETE' )
             END IF
           END IF
         END SELECT
       CASE ( 3 )                      ! Obtain the Jacobian
         nlp%J%val( 1 ) = mult
       CASE ( 4 )                      ! Obtain the Hessian
         nlp%H%val( 1 ) = 0.0_wp
       CASE ( 5 )                      ! form a Jacobian-vector product
         data%U( 1 ) = data%U( 1 ) + mult * data%V( 1 )
       CASE ( 6 )                      ! form a Hessian-vector product
         data%U( 1 ) = data%U( 1 )
       CASE ( 7 )               ! form residual Hessian-vector products
         nlp%P%val( 1 ) = 0.0_wp
       CASE ( 8 )                      ! Apply the preconditioner
         data%U( 1 ) = - data%V( 1 )
       CASE DEFAULT                    ! Terminal exit from loop
         EXIT
       END SELECT
     END DO
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I3, ':', I6, ' iterations. Optimal ||c|| = ', F6.1,        &
      &  ' status = ', I6 )" ) s, inform%iter, inform%norm_c, inform%status
     ELSE
       WRITE( 6, "( I3, ': NLS_solve exit status = ', I6 )" ) s, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
     CLOSE( UNIT = scratch_out )
   END DO

   control%subproblem_direct = .TRUE.         ! Use a direct method
   CALL NLS_solve( nlp, control, inform, data, userdata,                       &
                   eval_C = RES, eval_J = JAC, eval_H = HESS,                  &
                   eval_JPROD = JACPROD, eval_HPROD = HESSPROD,                &
                   eval_HPRODS  = RHESSPRODS )

   DEALLOCATE( nlp%X, nlp%C, nlp%G, userdata%real, W )
   DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
   DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
   IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )

!  =========================
!  test of available options
!  =========================

10 CONTINUE

!  IF ( .TRUE. ) GO TO 20
! start problem data
   nlp%n = 2 ;  nlp%m = 3 ; nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ), W( nlp%m ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne))
   nlp%J%row = (/ 1, 2, 2, 3, 3 /)              ! Jacobian J
   nlp%J%col = (/ 1, 1, 2, 1, 2 /)
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne))
   nlp%H%row = (/ 1, 2 /)                       ! Hessian H
   nlp%H%col = (/ 1, 2 /)                       ! NB lower triangle only
   ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ), nlp%P%ptr( nlp%m+1))
   nlp%P%row = (/ 1, 2 /)                       ! Hessian products
   nlp%P%ptr = (/ 1, 2, 3, 3 /)
   ALLOCATE( userdata%real( 1 ) )  ! Allocate space to hold parameter
   userdata%real( 1 ) = p          ! Record parameter, p
   W = 1.0_wp                      ! Record weight (if needed)
! problem data complete

   WRITE( 6, "( /, ' test of availible options ', / )" )

   DO i = 1, 17
     CALL NLS_initialize( data, control, inform ) ! Initialize control params
     control%jacobian_available = 2               ! the Jacobian is available
     control%hessian_available = 2                ! the Hessian is available
!    control%print_level = 1
     inform%status = 1                            ! set for initial entry
     nlp%X = 1.0_wp                               ! start from one

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
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 2 ) THEN
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 3 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 4 ) THEN
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 5 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%model = 5
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 6 ) THEN
       control%model = 5
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 7 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%model = 5
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 8 ) THEN
       control%model = 5
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 9 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%model = 5
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 10 ) THEN
       control%model = 5
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 11 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
       control%model = 5
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       control%subproblem_direct = .TRUE.         ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
       DEALLOCATE( nlp%VNAMES )
     ELSE IF ( i == 12 ) THEN
       control%model = 5
       control%norm = 2
       control%out = scratch_out
       control%error = scratch_out
       control%print_level = 101
       control%print_gap = 2
       control%stop_print = 5
       control%print_obj = .TRUE.
       control%psls_control%out = scratch_out
       control%psls_control%error = scratch_out
       control%psls_control%print_level = 1
       control%rqs_control%out = scratch_out
       control%rqs_control%error = scratch_out
       control%rqs_control%print_level = 1
       OPEN( UNIT = scratch_out, STATUS = 'SCRATCH' )
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 13 ) THEN
       control%norm = 3
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
     ELSE IF ( i == 14 ) THEN
       control%norm = 5
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
     ELSE IF ( i == 15 ) THEN
       control%norm = - 2
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
     ELSE IF ( i == 16 ) THEN
       control%model = 1
       control%maxit = 1000
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
     ELSE IF ( i == 17 ) THEN
       control%model = 3
       control%maxit = 1000
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,              &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,            &
                       eval_HPRODS  = RHESSPRODS )
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal ||c|| = ', F6.1,        &
      &  ' status = ', I6 )" ) i, inform%iter, inform%norm_c, inform%status
     ELSE
       WRITE( 6, "( I2, ': NLS_solve exit status = ', I6 )" ) i, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%C, nlp%G, W, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
   DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
   IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )

!  ============================================
!  test of scaling, model and weighting options
!  ============================================

20 CONTINUE
!  IF ( .TRUE. ) GO TO 30
! start problem data
   nlp%n = 2 ;  nlp%m = 3 ; nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ), W( nlp%m ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne))
   nlp%J%row = (/ 1, 2, 2, 3, 3 /)              ! Jacobian J
   nlp%J%col = (/ 1, 1, 2, 1, 2 /)
   ALLOCATE( userdata%real( 1 ) )  ! Allocate space to hold parameter
   userdata%real( 1 ) = p          ! Record parameter, p
   W = 1.0_wp                      ! Record weight (if needed)
! problem data complete

   WRITE( 6, "( /, ' test of scaling, model & weighting options ', / )" )

   DO model = 1, 8
!  DO model = 3, 3
     IF ( model == 4 ) THEN
     CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
       ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                 &
                 nlp%H%col( nlp%H%ne ) )
       nlp%H%row = (/ 1, 2 /) ! Hessian H
       nlp%H%col = (/ 1, 2 /) ! NB lower triangle only
     ELSE IF ( model == 6 ) THEN
       ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ),                 &
                 nlp%P%ptr( nlp%m + 1 ) )
       nlp%P%row = (/ 1, 2 /)  ! Hessian products
       nlp%P%ptr = (/ 1, 2, 3, 3 /)
     END IF
     DO scaling = - 1, 8
!    DO scaling = 1, 1
!    DO scaling = - 1, - 1
!      IF ( scaling == 0 .OR. scaling == 6 ) CYCLE
       DO rev = 0, 1
!      DO rev = 0, 0
         DO usew = 0, 1
!        DO usew = 0, 0
         CALL NLS_initialize( data, control, inform )! Initialize controls
         control%model = model             ! set model
         control%norm = scaling            ! set scaling norm
         control%jacobian_available = 2    ! the Jacobian is available
         IF ( model >= 4 ) control%hessian_available = 2 ! Hessian is available
!        control%print_level = 4
!        control%subproblem_control%print_level = 4
!        control%print_level = 4
!        control%maxit = 1
!        control%subproblem_control%print_level = 1
!        control%subproblem_control%magic_step = .TRUE.
!        control%subproblem_control%glrt_control%print_level = 3
         nlp%X = 1.0_wp                               ! start from one
         inform%status = 1                            ! set for initial entry
         IF ( rev == 0 ) THEN
           IF ( usew == 0 ) THEN
             CALL NLS_solve( nlp, control, inform, data, userdata,             &
                             eval_C = RES, eval_J = JAC, eval_H = HESS,        &
                             eval_JPROD = JACPROD, eval_HPROD = HESSPROD,      &
                             eval_HPRODS = RHESSPRODS )
           ELSE
             CALL NLS_solve( nlp, control, inform, data, userdata,             &
                             eval_C = RES, eval_J = JAC, eval_H = HESS,        &
                             eval_JPROD = JACPROD, eval_HPROD = HESSPROD,      &
                             eval_HPRODS = RHESSPRODS, W = W )
           END IF
         ELSE
           DO              ! Loop to solve problem
             IF ( usew == 0 ) THEN
               CALL NLS_solve( nlp, control, inform, data, userdata )
             ELSE
               CALL NLS_solve( nlp, control, inform, data, userdata, W = W )
             END IF
             SELECT CASE ( inform%status )   ! reverse communication
             CASE ( 2 )    ! Obtain the residuals
               CALL RES( data%eval_status, nlp%X, userdata, nlp%C )
             CASE ( 3 )    ! Obtain the Jacobian
               CALL JAC( data%eval_status, nlp%X, userdata, nlp%J%val )
             CASE ( 4 )    ! Obtain the Hessian
               CALL HESS( data%eval_status, nlp%X, data%Y, userdata,           &
                          nlp%H%val )
             CASE ( 5 )    ! form a Jacobian-vector product
               CALL JACPROD( data%eval_status, nlp%X, userdata,                &
                             data%transpose, data%U, data%V )
             CASE ( 6 )    ! form a Hessian-vector product
               CALL HESSPROD( data%eval_status, nlp%X, data%Y, userdata,       &
                              data%U, data%V )
             CASE ( 7 )    ! form residual Hessian-vector products
               CALL RHESSPRODS( data%eval_status, nlp%X, data%V, userdata,     &
                                nlp%P%val )
             CASE ( 8 )     ! Apply the preconditioner
               CALL SCALE( data%eval_status, nlp%X, userdata, data%U, data%V )
             CASE DEFAULT   ! Terminal exit from loop
               EXIT
             END SELECT
           END DO
         END IF
         IF ( inform%status == 0 ) THEN
           WRITE( 6, "( I1, ',', I2, 2( ',', I1 ), ':', I6, ' iterations.',    &
          &  ' Optimal objective value = ', F6.1, ' status = ', I6 )" )        &
            rev, scaling, model, usew, inform%iter, inform%norm_c, inform%status
         ELSE
           WRITE( 6, "( I1, ',', I2, 2( ',', I1 ), ': NLS_solve exit status',  &
          & ' = ', I6 )" ) rev, scaling, model, usew, inform%status
         END IF
         CALL NLS_terminate( data, control, inform ) ! delete workspace
       END DO
       END DO
     END DO
   END DO
30 CONTINUE
   DEALLOCATE( nlp%X, nlp%C, nlp%G, W, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
   DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
   IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )

!  ============================
!  full test of generic problem
!  ============================

!  IF ( .TRUE. ) GO TO 40

! start problem data
   nlp%n = 2 ;  nlp%m = 3 ; nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%J%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ), nlp%J%col( nlp%J%ne))
   nlp%J%row = (/ 1, 2, 2, 3, 3 /)              ! Jacobian J
   nlp%J%col = (/ 1, 1, 2, 1, 2 /)
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne))
   nlp%H%row = (/ 1, 2 /)                       ! Hessian H
   nlp%H%col = (/ 1, 2 /)                       ! NB lower triangle only
   ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ), nlp%P%ptr( nlp%m+1))
   nlp%P%row = (/ 1, 2 /)                       ! Hessian products
   nlp%P%ptr = (/ 1, 2, 3, 3 /)
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p
! problem data complete

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   DO i = 1, 6
!  DO i = 3, 3
     CALL NLS_initialize( data, control, inform ) ! Initialize control params
!    control%print_level = 1
!    control%print_level = 4
!    control%subproblem_control%print_level = 4
     control%jacobian_available = 2               ! the Jacobian is available
     control%hessian_available = 2                ! the Hessian is available
     inform%status = 1                            ! set for initial entry
     nlp%X = 1.0_wp                               ! start from one

     IF ( i == 1 ) THEN
       control%subproblem_direct = .TRUE.        ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,      &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,       &
                       eval_HPRODS = RHESSPRODS )
     ELSE IF ( i == 2 ) THEN
       control%hessian_available = 1     ! Hessian products will be used
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,      &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,       &
                       eval_HPRODS = RHESSPRODS )
     ELSE IF ( i == 3 ) THEN
       control%hessian_available = 1    ! Hessian products will be used
       control%norm = - 3               ! User's preconditioner
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = RES, eval_J = JAC, eval_H = HESS,      &
                       eval_JPROD = JACPROD, eval_HPROD = HESSPROD,       &
                       eval_HPRODS = RHESSPRODS, eval_SCALE = SCALE )
     ELSE IF ( i == 4 .OR. i == 5 .OR. i == 6 ) THEN
       IF ( i == 4 ) THEN
         control%subproblem_direct = .TRUE. ! Use a direct method
       ELSE
         control%hessian_available = 1   ! Hessian products will be used
       END IF
       IF ( i == 6 ) control%norm = - 3  ! User's scaling
       DO                                ! Loop to solve problem
         CALL NLS_solve( nlp, control, inform, data, userdata )
         SELECT CASE ( inform%status )   ! reverse communication
         CASE ( 2 )                      ! Obtain the residuals
           CALL RES( data%eval_status, nlp%X, userdata, nlp%C )
         CASE ( 3 )                      ! Obtain the Jacobian
           CALL JAC( data%eval_status, nlp%X, userdata, nlp%J%val )
         CASE ( 4 )                      ! Obtain the Hessian
           CALL HESS( data%eval_status, nlp%X, data%Y, userdata, data%H%val )
         CASE ( 5 )                      ! form a Jacobian-vector product
           CALL JACPROD( data%eval_status, nlp%X, userdata, data%transpose,    &
                         data%U, data%V )
         CASE ( 6 )                      ! form a Hessian-vector product
           CALL HESSPROD( data%eval_status, nlp%X, data%Y, userdata,           &
                          data%U, data%V )
         CASE ( 7 )                      ! form residual Hessian-vector products
           CALL RHESSPRODS( data%eval_status, nlp%X, data%V, userdata,         &
                            nlp%P%val )
         CASE ( 8 )                      ! Apply the preconditioner
           CALL SCALE( data%eval_status, nlp%X, userdata, data%U, data%V )
         CASE DEFAULT                    ! Terminal exit from loop
           EXIT
         END SELECT
       END DO
     ELSE
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal ||c|| = ', F6.1,        &
      &  ' status = ', I6 )" ) i, inform%iter, inform%norm_c, inform%status
     ELSE
       WRITE( 6, "( I2, ': NLS_solve exit status = ', I0 )" ) i, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%C, nlp%G, userdata%real )
   DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
   DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
   DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
   IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )

!  =======================
!  test of storage options
!  =======================

40 CONTINUE
   WRITE( 6, "( /, ' test of storage options', / )" )

! start problem data

   nlp%n = 2 ;  nlp%m = 3
   ALLOCATE( nlp%X( nlp%n ), nlp%C( nlp%m ), nlp%G( nlp%n ) )
   ALLOCATE( userdata%real( 1 ) ) ! Allocate space to hold parameter
   userdata%real( 1 ) = p         ! Record parameter, p

   DO store = 1, 3
!  DO store = 1, 1
     IF ( store == 1 ) THEN ! coordinate
       storage = 'C'
       nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
       IF ( ALLOCATED( nlp%J%type ) ) DEALLOCATE( nlp%J%type )
       CALL SMT_put( nlp%J%type, 'COORDINATE', s )
       ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%row( nlp%J%ne ),                 &
                 nlp%J%col( nlp%J%ne) )
       nlp%J%row = (/ 1, 2, 2, 3, 3 /)  ! Jacobian J
       nlp%J%col = (/ 1, 1, 2, 1, 2 /)
       IF ( ALLOCATED( nlp%H%type ) ) DEALLOCATE( nlp%H%type )
       CALL SMT_put( nlp%H%type, 'COORDINATE', s )
       ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ),                 &
                 nlp%H%col( nlp%H%ne) )
       nlp%H%row = (/ 1, 2 /)      ! Hessian H
       nlp%H%col = (/ 1, 2 /)      ! NB lower triangle only
       ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ),                 &
                 nlp%P%ptr( nlp%m+1))
       nlp%P%row = (/ 1, 2 /)       ! Hessian products
       nlp%P%ptr = (/ 1, 2, 3, 3 /)
     ELSE IF ( store == 2 ) THEN ! row wise
       storage = 'R'
       nlp%J%ne = 5 ; nlp%H%ne = 2 ; nlp%P%ne = 2
       IF ( ALLOCATED( nlp%J%type ) ) DEALLOCATE( nlp%J%type )
       CALL SMT_put( nlp%J%type, 'SPARSE_BY_ROWS', s )
       ALLOCATE( nlp%J%val( nlp%J%ne ), nlp%J%col( nlp%J%ne ),                 &
                 nlp%J%ptr( nlp%m + 1 ) )
       nlp%J%ptr = (/ 1, 2, 4, 6 /)  ! Jacobian J
       nlp%J%col = (/ 1, 1, 2, 1, 2 /)
       IF ( ALLOCATED( nlp%H%type ) ) DEALLOCATE( nlp%H%type )
       CALL SMT_put( nlp%H%type, 'SPARSE_BY_ROWS', s )
       ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%col( nlp%H%ne ),                 &
                 nlp%H%ptr( nlp%n + 1 ) )
       nlp%H%ptr = (/ 1, 2, 3 /)   ! Hessian H
       nlp%H%col = (/ 1, 2 /)      ! NB lower triangle only
       nlp%P%ne = 2
       ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ),                 &
                 nlp%P%ptr( nlp%m + 1 ) )
       nlp%P%row = (/ 1, 2 /)      ! Hessian products
       nlp%P%ptr = (/ 1, 2, 3, 3 /)
     ELSE IF ( store == 3 ) THEN ! dense
       storage = 'D'
!       nlp%J%m = nlp%m ; nlp%J%n = nlp%n
       IF ( ALLOCATED( nlp%J%type ) ) DEALLOCATE( nlp%J%type )
       CALL SMT_put( nlp%J%type, 'DENSE', s )
       ALLOCATE( nlp%J%val( nlp%m * nlp%n ) )
       IF ( ALLOCATED( nlp%H%type ) ) DEALLOCATE( nlp%H%type )
       CALL SMT_put( nlp%H%type, 'DENSE', s )
       ALLOCATE( nlp%H%val( nlp%n * ( nlp%n + 1 ) / 2 ) )
       IF ( ALLOCATED( nlp%P%type ) ) DEALLOCATE( nlp%P%type )
       CALL SMT_put( nlp%P%type, 'DENSE_BY_COLUMNS', s )
       ALLOCATE( nlp%P%val( nlp%m * nlp%n ) )
     END IF

! problem data complete

     DO i = 1, 2
!    DO i = 2, 2
       CALL NLS_initialize( data, control, inform ) ! Initialize control params
       control%jacobian_available = 2               ! the Jacobian is available
       control%hessian_available = 2                ! the Hessian is available
       control%subproblem_direct = .TRUE. ! Use a direct method
!      control%subproblem_control%subproblem_direct = .TRUE.
       control%model = 6   ! tensor-Gauss-Newton model
       control%norm = - 3  ! User's scaling
       control%norm = - 1  ! Euclidean scaling
!      control%print_level = 1
!      control%print_level = 4
!      control%subproblem_control%print_level = 1
       inform%status = 1                            ! set for initial entry
       nlp%X = 1.0_wp                               ! start from one

       IF ( i == 1 ) THEN
         IF ( store == 3 ) THEN
           CALL NLS_solve( nlp, control, inform, data, userdata,               &
                           eval_C = RES, eval_J = JAC_dense,                   &
                           eval_H = HESS_dense,                                &
                           eval_HPRODS = RHESSPRODS_dense, eval_SCALE = SCALE )
         ELSE
           CALL NLS_solve( nlp, control, inform, data, userdata,               &
                           eval_C = RES, eval_J = JAC, eval_H = HESS,          &
                           eval_HPRODS = RHESSPRODS, eval_SCALE = SCALE )
         END IF
       ELSE
         DO                              ! Loop to solve problem
           CALL NLS_solve( nlp, control, inform, data, userdata )
           SELECT CASE ( inform%status ) ! reverse communication
           CASE ( 2 )                    ! Obtain the residuals
             CALL RES( data%eval_status, nlp%X, userdata, nlp%C )
           CASE ( 3 )                    ! Obtain the Jacobian
             IF ( store == 3 ) THEN
               CALL JAC_dense( data%eval_status, nlp%X, userdata, nlp%J%val )
             ELSE
               CALL JAC( data%eval_status, nlp%X, userdata, nlp%J%val )
             END IF
           CASE ( 4 )                    ! Obtain the Hessian
             IF ( store == 3 ) THEN
               CALL HESS_dense( data%eval_status, nlp%X, data%Y, userdata,     &
                                data%H%val )
             ELSE
               CALL HESS( data%eval_status, nlp%X, data%Y, userdata,           &
                          data%H%val )
             END IF
           CASE ( 5 )                    ! form a Jacobian-vector product
             CALL JACPROD( data%eval_status, nlp%X, userdata, data%transpose,  &
                           data%U, data%V )
           CASE ( 6 )                    ! form a Hessian-vector product
             CALL HESSPROD( data%eval_status, nlp%X, data%Y, userdata,         &
                            data%U, data%V )
           CASE ( 7 )                 ! form residual Hessian-vector products
             IF ( store == 3 ) THEN
               CALL RHESSPRODS_dense( data%eval_status, nlp%X, data%V,         &
                                      userdata, nlp%P%val )
             ELSE
               CALL RHESSPRODS( data%eval_status, nlp%X, data%V, userdata,     &
                                nlp%P%val )
             END IF
           CASE ( 8 )                      ! Apply the preconditioner
             CALL SCALE( data%eval_status, nlp%X, userdata, data%U, data%V )
           CASE DEFAULT                    ! Terminal exit from loop
             EXIT
           END SELECT
         END DO
       END IF
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( A1, I1, ':', I6, ' iterations. Optimal ||c|| = ', F6.1,  &
           &  ' status = ', I6 )" )                                            &
           storage, i, inform%iter, inform%norm_c, inform%status
       ELSE
         WRITE( 6, "( A1, I1, ': NLS_solve exit status = ', I0 )" )            &
           storage, i, inform%status
       END IF
       CALL NLS_terminate( data, control, inform )  ! delete internal workspace
     END DO
     IF ( store == 1 ) THEN ! coordinate
       DEALLOCATE( nlp%J%val, nlp%J%row, nlp%J%col, nlp%J%type )
       DEALLOCATE( nlp%H%val, nlp%H%row, nlp%H%col, nlp%H%type )
       DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
     ELSE IF ( store == 2 ) THEN ! row wise
       DEALLOCATE( nlp%J%val, nlp%J%col, nlp%J%ptr, nlp%J%type )
       DEALLOCATE( nlp%H%val, nlp%H%col, nlp%H%ptr, nlp%H%type )
       DEALLOCATE( nlp%P%val, nlp%P%row, nlp%P%ptr )
     ELSE IF ( store == 3 ) THEN ! dense
       DEALLOCATE( nlp%J%val, nlp%J%type )
       DEALLOCATE( nlp%H%val, nlp%H%type )
       DEALLOCATE( nlp%P%val, nlp%P%type )
     END IF
   END DO

   DEALLOCATE( nlp%X, nlp%C, nlp%G, userdata%real )

   CONTAINS

     SUBROUTINE RES( status, X, userdata, C )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     C( 1 ) = X( 1 ) ** 2 + userdata%real( 1 )
     C( 2 ) = X( 1 ) + X( 2 ) ** 2
     C( 3 ) = X( 1 ) - X( 2 )
     status = 0
     END SUBROUTINE RES

     SUBROUTINE JAC( status, X, userdata, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     J_val( 1 ) = 2.0_wp * X( 1 )
     J_val( 2 ) = 1.0_wp
     J_val( 3 ) = 2.0_wp * X( 2 )
     J_val( 4 ) = 1.0_wp
     J_val( 5 ) = - 1.0_wp
     status = 0
     END SUBROUTINE JAC

     SUBROUTINE HESS( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_wp * Y( 1 )
     H_val( 2 ) = 2.0_wp * Y( 2 )
     status = 0
     END SUBROUTINE HESS

     SUBROUTINE JACPROD( status, X, userdata, transpose, U, V, got_j )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j
     IF ( transpose ) THEN
       U( 1 ) = U( 1 ) + 2.0_wp * X( 1 ) * V( 1 ) + V( 2 ) + V( 3 )
       U( 2 ) = U( 2 ) + 2.0_wp * X( 2 ) * V( 2 ) - V( 3 )
     ELSE
       U( 1 ) = U( 1 ) + 2.0_wp * X( 1 ) * V( 1 )
       U( 2 ) = U( 2 ) + V( 1 )  + 2.0_wp * X( 2 ) * V( 2 )
       U( 3 ) = U( 3 ) + V( 1 ) - V( 2 )
     END IF
     status = 0
     END SUBROUTINE JACPROD

     SUBROUTINE HESSPROD( status, X, Y, userdata, U, V, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     U( 1 ) = U( 1 ) + 2.0_wp * Y( 1 ) * V( 1 )
     U( 2 ) = U( 2 ) + 2.0_wp * Y( 2 ) * V( 2 )
     status = 0
     END SUBROUTINE HESSPROD

     SUBROUTINE RHESSPRODS( status, X, V, userdata, P_val, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     P_val( 1 ) = 2.0_wp * V( 1 )
     P_val( 2 ) = 2.0_wp * V( 2 )
     status = 0
     END SUBROUTINE RHESSPRODS

     SUBROUTINE SCALE( status, X, userdata, U, V )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
!     U( 1 ) = 0.5_wp * V( 1 )
!     U( 2 ) = 0.5_wp * V( 2 )
     U( 1 ) = V( 1 )
     U( 2 ) = V( 2 )
     status = 0
     END SUBROUTINE SCALE

     SUBROUTINE JAC_dense( status, X, userdata, J_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     J_val( 1 ) = 2.0_wp * X( 1 )
     J_val( 2 ) = 0.0_wp
     J_val( 3 ) = 1.0_wp
     J_val( 4 ) = 2.0_wp * X( 2 )
     J_val( 5 ) = 1.0_wp
     J_val( 6 ) = - 1.0_wp
     status = 0
     END SUBROUTINE JAC_dense

     SUBROUTINE HESS_dense( status, X, Y, userdata, H_val )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     H_val( 1 ) = 2.0_wp * Y( 1 )
     H_val( 2 ) = 0.0_wp
     H_val( 3 ) = 2.0_wp * Y( 2 )
     status = 0
     END SUBROUTINE HESS_dense

     SUBROUTINE RHESSPRODS_dense( status, X, V, userdata, P_val, got_h )
     USE GALAHAD_USERDATA_double
     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
     P_val( 1 ) = 2.0_wp * V( 1 )
     P_val( 2 ) = 0.0_wp
     P_val( 3 ) = 0.0_wp
     P_val( 4 ) = 2.0_wp * V( 2 )
     P_val( 5 ) = 0.0_wp
     P_val( 6 ) = 0.0_wp
     status = 0
     END SUBROUTINE RHESSPRODS_dense

   END PROGRAM GALAHAD_NLS_test_deck
