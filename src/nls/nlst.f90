! THIS VERSION: GALAHAD 2.7 - 27/10/2015 AT 13:15 GMT
   PROGRAM GALAHAD_NLS_test_deck
   USE GALAHAD_NLS_double                       ! double precision version
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( NLS_control_type ) :: control
   TYPE ( NLS_inform_type ) :: inform
   TYPE ( NLS_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   REAL ( KIND = wp ), PARAMETER :: p = 1.0_wp
   EXTERNAL :: eval_C, eval_J, eval_H, eval_JPROD, eval_HPROD, eval_HPRODS
   INTEGER :: i, s, model, scaling
   INTEGER :: scratch_out = 56
   logical :: alive
   REAL ( KIND = wp ) :: dum

   GO TO 10

! start problem data
   nlp%n = 1 ; nlp%H%ne = 1                     ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne))
   nlp%H%row = (/ 1 /) ; nlp%H%col = (/ 1 /)
   ALLOCATE( userdata%real( 1 ) )               ! Allocate space for parameter
   userdata%real( 1 ) = p                       ! Record parameter, p
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
     IF ( s == - GALAHAD_error_unbounded ) CYCLE
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
     CALL NLS_initialize( data, control,inform ) ! Initialize control params
!     control%print_level = 4
!     control%RQS_control%print_level = 4
!     control%GLTR_control%print_level = 4
     inform%status = 1                            ! set for initial entry
     nlp%n = 1
     nlp%X = 1.0_wp                               ! start from one
     control%hessian_available = 1                ! Hessian prods will be used

     IF ( s == - GALAHAD_error_restrictions ) THEN
       nlp%n = 0
     ELSE IF ( s == - GALAHAD_error_preconditioner ) THEN
       control%norm = - 3               ! User's preconditioner
     ELSE IF ( s == - GALAHAD_error_max_iterations ) THEN
       control%maxit = 0
     ELSE IF ( s == - GALAHAD_error_cpu_limit ) THEN
       control%cpu_time_limit = 0.0_wp
     END IF
     DO                                           ! Loop to solve problem
!      write(6,*) 'in ', inform%status
       CALL NLS_solve( nlp, control, inform, data, userdata )
!      write(6,*) 'out ', inform%status
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
       WRITE( 6, "( I2, ': NLS_solve exit status = ', I6 )" ) s, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

   control%subproblem_direct = .TRUE.         ! Use a direct method
   CALL NLS_solve( nlp, control, inform, data, userdata,                       &
                   eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,          &
                   eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,           &
                   eval_HPRODS  = eval_HPRODS )

   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col )

!  =========================
!  test of available options
!  =========================

10 CONTINUE

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
   CALL SMT_put( nlp%P%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%P%val( nlp%P%ne ), nlp%P%row( nlp%P%ne ), nlp%P%ptr( nlp%m+1))
   nlp%P%row = (/ 1, 2 /)                       ! Hessian products
   nlp%P%ptr = (/ 1, 2, 3, 3 /)
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p
! problem data complete
   IF ( .TRUE. ) GO TO 20

   WRITE( 6, "( /, ' test of availible options ', / )" )

   DO i = 1, 7
     CALL NLS_initialize( data, control, inform )! Initialize control params
!    control%print_level = 1
     inform%status = 1                            ! set for initial entry
     nlp%X = 1.0_wp                               ! start from one

     IF ( i == 1 ) THEN
       ALLOCATE( nlp%VNAMES( nlp%n ) )
       nlp%VNAMES( 1 ) = 'X1' ; nlp%VNAMES( 1 ) = 'X2' ; nlp%VNAMES( 1 ) = 'X3'
!      control%out = 6
!      control%error = 6
!      control%print_level = 1
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
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
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
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
       CLOSE( UNIT = scratch_out )
     ELSE IF ( i == 3 ) THEN
       control%norm = 3
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 4 ) THEN
       control%norm = 5
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 5 ) THEN
       control%norm = - 2
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 6 ) THEN
       control%model = 1
       control%maxit = 1000
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 7 ) THEN
       control%model = 3
       control%maxit = 1000
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': NLS_solve exit status = ', I6 )" ) i, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO

20 CONTINUE
   WRITE( 6, "( /, ' test of scaling,model options ', / )" )

!  DO scaling = - 1, 8
   DO scaling = 2, 2
     IF ( scaling == 0 .OR. scaling == 6 ) CYCLE
!    DO model = 1, 8
!    DO model = 6, 6
     DO model = 7, 7
       CALL NLS_initialize( data, control, inform )! Initialize control params
       control%model = model                        ! set model
       control%norm = scaling                       ! set scaling norm
       nlp%X = 1.0_wp                               ! start from one
       inform%status = 1                            ! set for initial entry
       control%print_level = 4
       control%subproblem_control%print_level = 1
!      control%subproblem_control%magic_step = .TRUE.
!      control%subproblem_control%glrt_control%print_level = 3
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( I2, ',', I1, ':', I6, ' iterations. Optimal objective ', &
        &  'value = ', F6.1, ' status = ', I6 )" )                             &
           scaling, model, inform%iter, inform%obj, inform%status
       ELSE
         WRITE( 6, "( I2, ',', I1, ': NLS_solve exit status = ', I6 )" )       &
           scaling, model, inform%status
       END IF
     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
     END DO
   END DO

   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
stop

!  ============================
!  full test of generic problem
!  ============================

! start problem data
   nlp%n = 3 ; nlp%H%ne = 5                  ! dimensions
   ALLOCATE( nlp%X( nlp%n ), nlp%G( nlp%n ) )
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( nlp%H%ne ), nlp%H%row( nlp%H%ne ), nlp%H%col( nlp%H%ne))
   nlp%H%row = (/ 1, 3, 2, 3, 3 /)              ! Hessian H
   nlp%H%col = (/ 1, 1, 2, 2, 3 /)              ! NB lower triangle
! problem data complete
   ALLOCATE( userdata%real( 1 ) )             ! Allocate space to hold parameter
   userdata%real( 1 ) = p                     ! Record parameter, p

   WRITE( 6, "( /, ' full test of generic problems ', / )" )

   DO i = 1, 3
     CALL NLS_initialize( data, control, inform )! Initialize control params
!    control%print_level = 1
     inform%status = 1                            ! set for initial entry
     nlp%X = 1.0_wp                               ! start from one

     IF ( i == 1 ) THEN
       control%subproblem_direct = .TRUE.        ! Use a direct method
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 2 ) THEN
       control%hessian_available = 1             ! Hessian products will be used
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 3 ) THEN
       control%hessian_available = 1            ! Hessian products will be used
       control%norm = - 3               ! User's preconditioner
       CALL NLS_solve( nlp, control, inform, data, userdata,                   &
                       eval_C = eval_C, eval_J = eval_J, eval_H = eval_H,      &
                       eval_JPROD = eval_JPROD, eval_HPROD = eval_HPROD,       &
                       eval_HPRODS  = eval_HPRODS )
     ELSE IF ( i == 4 .OR. i == 5 .OR. i == 6 ) THEN
       IF ( i == 4 ) THEN
         control%subproblem_direct = .TRUE.         ! Use a direct method
       ELSE
         control%hessian_available = 1              ! Hessian prods will be used
       END IF
       IF ( i == 6 ) control%norm = - 3   ! User's preconditioner
       DO                                           ! Loop to solve problem
         CALL NLS_solve( nlp, control, inform, data, userdata )
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
           data%eval_status = 0                   ! record successful evaluation
         CASE ( 5 )                               ! Obtain Hessian-vector prod
           data%U( 1 ) = data%U( 1 ) + 2.0_wp * ( data%V( 1 ) + data%V( 3 ) )  &
                         - COS( nlp%X( 1 ) ) * data%V( 1 )
           data%U( 2 ) = data%U( 2 ) + 2.0_wp * ( data%V( 2 ) + data%V( 3 ) )
           data%eval_status = 0                   ! record successful evaluation
         CASE ( 6 )                               ! Apply the preconditioner
           data%U( 1 ) = 0.5_wp * data%V( 1 )
           data%U( 2 ) = 0.5_wp * data%V( 2 )
           data%eval_status = 0                   ! record successful evaluation
         CASE DEFAULT                             ! Terminal exit from loop
           EXIT
         END SELECT
       END DO
     ELSE
     END IF
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( I2, ':', I6, ' iterations. Optimal objective value = ',    &
     &     F6.1, ' status = ', I6 )" ) i, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( I2, ': NLS_solve exit status = ', I6 )" ) i, inform%status
     END IF

     CALL NLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col, userdata%real )
   END PROGRAM GALAHAD_NLS_test_deck

   SUBROUTINE eval_C( status, X, userdata, C )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: C
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   C( 1 ) = X( 1 ) ** 2 + userdata%real( 1 )
   C( 2 ) = X( 1 ) + X( 2 ) ** 2
   C( 3 ) = X( 1 ) - X( 2 )
   status = 0
   END SUBROUTINE eval_C

   SUBROUTINE eval_J( status, X, userdata, J_val )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: J_val
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   J_val( 1 ) = 2.0_wp * X( 1 )
   J_val( 2 ) = 1.0_wp
   J_val( 3 ) = 2.0_wp * X( 2 )
   J_val( 4 ) = 1.0_wp
   J_val( 5 ) = - 1.0_wp
   status = 0
   END SUBROUTINE eval_J

   SUBROUTINE eval_H( status, X, Y, userdata, H_val )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   H_val( 1 ) = 2.0_wp * Y( 1 )
   H_val( 2 ) = 2.0_wp * Y( 2 )
   status = 0
   END SUBROUTINE eval_H

   SUBROUTINE eval_JPROD( status, X, userdata, transpose, U, V, got_j )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
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
   END SUBROUTINE eval_JPROD

   SUBROUTINE eval_HPROD( status, X, Y, userdata, U, V, got_h )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   U( 1 ) = U( 1 ) + 2.0_wp * Y( 1 ) * V( 1 )
   U( 2 ) = U( 2 ) + 2.0_wp * Y( 2 ) * V( 2 )
   status = 0
   END SUBROUTINE eval_HPROD

   SUBROUTINE eval_HPRODS( status, X, V, userdata, P_val, got_h )
   USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
   P_val( 1 ) = 2.0_wp * V( 1 )
   P_val( 2 ) = 2.0_wp * V( 2 )
   status = 0
   END SUBROUTINE eval_HPRODS
