! THIS VERSION: GALAHAD 5.5 - 2026-01-19 AT 10:30 GMT.
   PROGRAM GALAHAD_SLLSB_EXAMPLE3
   USE GALAHAD_SLLSB_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLSB_data_type ) :: data
   TYPE ( SLLSB_control_type ) :: control
   TYPE ( SLLSB_inform_type ) :: inform
   INTEGER :: s
   INTEGER, PARAMETER :: m = 3
!  INTEGER, PARAMETER :: n = 6
   INTEGER, PARAMETER :: n = 10002
   INTEGER, PARAMETER :: o = n + 1, ao_ne = 2 * n + 2, a_ne = n - 1
   REAL ( KIND = wp ), DIMENSION( o ) :: W
! start problem data
   ALLOCATE( p%B( o ), p%R( o ) )
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%o = o ; p%m = m ;              ! dimensions
   p%B( : n ) = 10.0_wp ; p%B( o ) = -10.0_wp ! observations
   p%C_l = 1.0_wp       ! constraint lower bound
   p%C_u = 1.0_wp       ! constraint upper bound
   p%X_l = 0.0_wp, p%X_l( n ) = - infinity ! variable lower bound
   p%X_u = infinity     ! variable upper bound
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero
!  set Ao
   CALL SMT_put( p%Ao%type, 'COORDINATE', s )     ! Co-ordinate storage for Ao
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   p%Ao%m = o ; p%Ao%n = n ; p%Ao%ne = 0
   DO j = 1, n
     p%Ao%ne = p%Ao%ne + 1
     p%Ao%row( p%Ao%ne ) = j
     p%Ao%col( p%Ao%ne ) = j
     p%Ao%val( p%Ao%ne ) = 1.0_wp
     p%Ao%ne = p%Ao%ne + 1
     p%Ao%row( p%Ao%ne ) = j + 1
     p%Ao%col( p%Ao%ne ) = j
     p%Ao%val( p%Ao%ne ) = - 1.0_wp
   END DO
!  set A
   CALL SMT_put( p%A%type, 'COORDINATE', s )     ! Co-ordinate storage for A
   ALLOCATE( p%Ao%val( ao_ne ), p%Ao%row( ao_ne ), p%Ao%col( ao_ne ) )
   p%A%m = m ; p%A%n = n ; p%A%ne = 0
   DO j = 1, n, m
     p%A%ne = p%Ao%ne + 1
     p%A%row( p%Ao%ne ) = 1
     p%A%col( p%Ao%ne ) = j
     p%A%val( p%Ao%ne ) = 1.0_wp
     IF ( j + 1 < n ) THEN
       p%A%ne = p%Ao%ne + 1
       p%A%row( p%Ao%ne ) = 2
       p%A%col( p%Ao%ne ) = j + 1
       p%A%val( p%Ao%ne ) = 1.0_wp
     END IF
     IF ( j + 2 < n ) THEN
       p%A%ne = p%Ao%ne + 1
       p%A%row( p%Ao%ne ) = 3
       p%A%col( p%Ao%ne ) = j + 2
       p%A%val( p%Ao%ne ) = 1.0_wp
     END IF
   END DO
! problem data complete
   CALL SLLSB_initialize( data, control, inform ) ! Initialize control parameters
!  control%print_level = 1
   control%infinity = infinity                 ! Set infinity
   CALL SLLSB_solve( p, data, control, inform ) ! Solve
   IF ( inform%status == 0 ) THEN              !  Successful return
     WRITE( 6, "( ' SLLSB: ', I0, ' iterations  ', /,                           &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
   ELSE                                         !  Error returns
     WRITE( 6, "( ' SLLSB_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL SLLSB_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_SLLSB_EXAMPLE3
