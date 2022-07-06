! THIS VERSION: GALAHAD 4.1 - 2022-05-26 AT 07:15 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE
   USE GALAHAD_SLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, m = 4, a_ne = 5
! start problem data
   ALLOCATE( p%B( m ), p%X( n ), X_stat( n ) )
   p%n = n ; p%m = m                          ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! Co-ordinate  storage for A
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( A_ne ) )
   p%A%m = m ; p%A%n = n
   p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   p%A%row = (/ 1, 2, 2, 3, 4 /)                     !
   p%A%col = (/ 1, 1, 2, 3, 3 /) ; p%A%ne = a_ne
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
!  control%CONVERT_control%print_level = 3
   inform%status = 1
   CALL SLLS_solve( p, X_stat, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
   END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Z, X_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type )
   END PROGRAM GALAHAD_SLLS_EXAMPLE
