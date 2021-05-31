! THIS VERSION: GALAHAD 3.3 - 29/05/2021 AT 14:00 GMT.
   PROGRAM GALAHAD_PRESOLVE_TEST  ! needs to be improved !!
   USE GALAHAD_QPT_double                           ! Double precision
   USE GALAHAD_PRESOLVE_double                      ! Double precision
   USE GALAHAD_SYMBOLS                              ! The GALAHAD symbols
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D0 )         ! Set precision
   REAL ( KIND = wp ), PARAMETER  :: infinity = 10._wp ** 20
   REAL ( KIND = wp ), PARAMETER  :: r0 = 0.0_wp, r1 = 1.0_wp
   REAL ( KIND = wp ), PARAMETER  :: r2 = 2.0_wp, r3 = 3.0_wp
   TYPE ( QPT_problem_type )      :: problem
   TYPE ( PRESOLVE_control_type ) :: control
   TYPE ( PRESOLVE_inform_type )  :: inform
   TYPE ( PRESOLVE_data_type )    :: data
   INTEGER :: j, n, m, h_ne, a_ne, s
!
!                                                +-----------------------------+
   INTEGER :: data_storage_type =  0             ! change this to -2 for dense !
                                                 ! or to -1 for row-wise sparse!
!                                                +-----------------------------+
! start problem data
   n = 6; m = 5; h_ne = 1; a_ne = 8
   problem%new_problem_structure = .TRUE.
   problem%n   = n; problem%m = m; problem%f = r1
   ALLOCATE( problem%G( n )  , problem%X_l( n ), problem%X_u( n ) )
   ALLOCATE( problem%C_l( m ), problem%C_u( m ) )
   problem%gradient_kind = 1
   problem%C_l = (/  r0, r0, r2, r1, r3  /)
   problem%C_u = (/  r1, r1, r3, r3, r3  /)
   problem%X_l = (/ -r3, r0, r0, r0, r0, r0 /)
   problem%X_u = (/  r3, r1, r1, r1, r1, r1 /)
! sparse coordinate format
   IF ( data_storage_type == 0 ) THEN
     CALL SMT_put( problem%H%type, 'COORDINATE', s )
     CALL SMT_put( problem%A%type, 'COORDINATE', s )
     ALLOCATE( problem%H%val( h_ne ) )
     ALLOCATE( problem%H%col( h_ne ), problem%H%row( h_ne ) )
     ALLOCATE( problem%A%val( a_ne ) )
     ALLOCATE( problem%A%col( a_ne ), problem%A%row( a_ne ) )
     problem%H%val = (/ r1  /)
     problem%H%row = (/ 1   /)
     problem%H%col = (/ 1   /)
     problem%A%val = (/ r1, r1, r1, r1, r1, r1, r1, r1 /)
     problem%A%row = (/  3,  3,  3,  4,  4,  5,  5,  5 /)
     problem%A%col = (/  3,  4,  5,  3,  6,  4,  5,  6 /)
     problem%A%ne  = a_ne; problem%H%ne = h_ne
! sparse row-wise storage format
   ELSE IF ( data_storage_type == - 1 ) THEN
     CALL SMT_put( problem%H%type, 'SPARSE_BY_ROWS', s )
     CALL SMT_put( problem%A%type, 'SPARSE_BY_ROWS', s )
     ALLOCATE( problem%H%val( h_ne ), problem%H%col( h_ne ) )
     ALLOCATE( problem%H%ptr( n+1 ) )
     ALLOCATE( problem%A%val( a_ne ), problem%A%col( a_ne ) )
     ALLOCATE( problem%A%ptr( m+1 ) )
     problem%H%val = (/ r1  /)
     problem%H%ptr = (/  1,  2,  2,  2,  2,  2,  2 /)
     problem%H%col = (/  1  /)
     problem%A%val = (/ r1, r1, r1, r1, r1, r1, r1, r1 /)
     problem%A%ptr = (/  1,  1,  1,  4,  6,  9 /)
     problem%A%col = (/  3,  4,  5,  3,  6,  4,  5,  6 /)
! dense storage format
   ELSE IF ( data_storage_type == - 2 ) THEN
     CALL SMT_put( problem%H%type, 'DENSE', s )
     CALL SMT_put( problem%A%type, 'DENSE', s )
     ALLOCATE( problem%H%val( n*(n+1)/2 ) )
     ALLOCATE( problem%A%val( n*m ) )
     problem%H%val = (/ r1,                           &
                        r0, r0,                       &
                        r0, r0, r0,                   &
                        r0, r0, r0, r0,               &
                        r0, r0, r0, r0, r0,           &
                        r0, r0, r0, r0, r0, r0  /)
     problem%A%val = (/ r0, r0, r0, r0, r0, r0,       &
                        r0, r0, r0, r0, r0, r0,       &
                        r0, r0, r1, r1, r1, r0,       &
                        r0, r0, r1, r0, r0, r1,       &
                        r0, r0, r0, r1, r1, r1  /)
   END IF
! problem data complete
! write the original formulation
   CALL QPT_write_problem( 6, problem ) 
! set the default PRESOLVE control parameters            
   CALL PRESOLVE_initialize( control, inform, data )
   IF ( inform%status /= 0 ) STOP
   control%print_level = - 1
   control%out = 0
   control%errout = 0
! apply presolving to reduce the problem
   CALL PRESOLVE_apply( problem, control, inform, data )
   IF ( inform%status /= 0 ) STOP
! write the reduced problem
   CALL QPT_write_problem( 6, problem ) 
! solve the reduced problem
   ! CALL QPSOLVER (unnecessary here, because the reduced problem has a 
   ! single feasible point in this example)
! restore the solved reduced problem to the original formulation
   CALL PRESOLVE_restore( problem, control, inform, data )
   IF ( inform%status /= 0 ) STOP
! write the final solution in the original variables
   WRITE( 6, * ) '  The problem solution X is'
   DO j = 1, n
      WRITE( 6, '(3x,''x('',I1,'') = '', ES12.4)' ) j, problem%X( j )
   END DO
! deallocate internal workspace
   CALL PRESOLVE_terminate( control, inform, data ) ! Delete internal workspace
   WRITE( 6, "( /, ' far from comprehensive !!' )" )
   END PROGRAM GALAHAD_PRESOLVE_TEST
