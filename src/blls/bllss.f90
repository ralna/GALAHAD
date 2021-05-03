! THIS VERSION: GALAHAD 3.3 - 11/12/2020 AT 15:50 GMT.
   PROGRAM GALAHAD_BLLS_EXAMPLE
   USE GALAHAD_BLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BLLS_data_type ) :: data
   TYPE ( BLLS_control_type ) :: control
   TYPE ( BLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, m = 4, a_ne = 5
! start problem data
   ALLOCATE( p%B( m ), p%X_l( n ), p%X_u( n ), p%X( n ), X_stat( n ) )
   p%n = n ; p%m = m                          ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X_l = (/ - 1.0_wp, - infinity, 0.0_wp /) ! variable lower bound
   p%X_u = (/ infinity, 1.0_wp, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! Co-ordinate  storage for A
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( A_ne ) )
   p%A%m = m ; p%A%n = n
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 2, 2, 3, 4 /)                     !
   p%A%col = (/ 1, 1, 2, 3, 3 /) ; p%A%ne = a_ne
! problem data complete
   CALL BLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
   control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
!  control%CONVERT_control%print_level = 3
   inform%status = 1
   CALL BLLS_solve( p, X_stat, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' BLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' BLLS_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
   END IF
   CALL BLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%X_l, p%X_u, p%Z, X_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
   END PROGRAM GALAHAD_BLLS_EXAMPLE
