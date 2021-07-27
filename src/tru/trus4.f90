   PROGRAM GALAHAD_TRU4_EXAMPLE  !  GALAHAD 3.3 - 25/07/2021 AT 09:15 GMT
   USE GALAHAD_TRU_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( TRU_control_type ) :: control
   TYPE ( TRU_inform_type ) :: inform
   TYPE ( TRU_full_data_type ) :: data
   TYPE ( NLPT_userdata_type ) :: userdata
   REAL ( KIND = wp ) :: f
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row, H_col, H_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, G, U, V
   EXTERNAL :: FUN, GRAD, HESS
   INTEGER :: s, status, eval_status, ne
   INTEGER, PARAMETER :: n = 3
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp
! start problem data
   ALLOCATE( X( n ), G( n ), U( n ), V( n ) )
   ALLOCATE( H_row( 0 ), H_col( 0 ), H_ptr( 0 ) )
   X = 1.0_wp                               ! start from one
! problem data complete   
   CALL TRU_initialize( data, control, inform ) ! Initialize control parameters
   control%hessian_available = .FALSE.          ! Hessian products will be used
!  control%print_level = 5
!  control%psls_control%preconditioner = - 3    ! Apply uesr's preconditioner
   CALL TRU_import( control, data, status, n,                                  &
                    'absent', ne, H_row, H_col, H_ptr )
   status = 1                                   ! Set for initial entry
   DO                                           ! Loop to solve problem
     CALL TRU_solve_reverse_without_h( data, status, eval_status,              &
                                       X, f, G, U, V )
     SELECT CASE ( status )                     ! reverse communication
     CASE ( 2 )                                 ! Obtain the objective function
       f = ( X( 1 ) + X( 3 ) + p ) ** 2 + ( X( 2 ) + X( 3 ) ) ** 2             &
              + COS( X( 1 ) )
       eval_status = 0                     ! record successful evaluation
     CASE ( 3 )                                 ! Obtain the gradient
       G( 1 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + p ) - SIN( X( 1 ) )
       G( 2 ) = 2.0_wp * ( X( 2 ) + X( 3 ) )
       G( 3 ) = 2.0_wp * ( X( 1 ) + X( 3 ) + p ) + 2.0_wp * ( X( 2 ) + X( 3 ) )
       eval_status = 0                     ! record successful evaluation
     CASE ( 5 )                            ! Obtain Hessian-vector product
       U( 1 ) = U( 1 ) + 2.0_wp * ( V( 1 ) + V( 3 ) ) - COS( X( 1 ) ) * V( 1 )
       U( 2 ) = U( 2 ) + 2.0_wp * ( V( 2 ) + V( 3 ) )
       U( 3 ) = U( 3 ) + 2.0_wp * ( V( 1 ) + V( 2 ) + 2.0_wp * V( 3 ) )
       eval_status = 0                     ! record successful evaluation
     CASE ( 6 )                            ! Apply the preconditioner
       U( 1 ) = 0.5_wp * V( 1 )
       U( 2 ) = 0.5_wp * V( 2 )
       U( 3 ) = 0.25_wp * V( 3 )
       eval_status = 0                     ! record successful evaluation
     CASE DEFAULT                          ! Terminal exit from loop
       EXIT
     END SELECT
   END DO
   CALL TRU_information( data, inform, status )
   IF ( inform%status == 0 ) THEN          ! Successful return
     WRITE( 6, "( ' TRU: ', I0, ' iterations -',                               &
    &     ' optimal objective value =',                                        &
    &       ES12.4, ', ||g|| =', ES11.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, inform%norm_g, X
   ELSE                                    ! Error returns
     WRITE( 6, "( ' TRU_solve exit status = ', I6 ) " ) inform%status
   END IF
   DEALLOCATE( X, G, U, V, H_row, H_col, H_ptr )
   CALL TRU_terminate( data, control, inform )  ! Delete internal workspace
   END PROGRAM GALAHAD_TRU4_EXAMPLE
