! THIS VERSION: GALAHAD 5.5 - 2026-01-19 AT 10:30 GMT.
   PROGRAM GALAHAD_SLLSB_EXAMPLE2
   USE GALAHAD_SLLSB_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLSB_data_type ) :: data
   TYPE ( SLLSB_control_type ) :: control
   TYPE ( SLLSB_inform_type ) :: inform
   INTEGER :: s
   INTEGER, PARAMETER :: n = 3, o = 4, m = 2, ao_ne = 7
! start problem data
   ALLOCATE( p%B( o ), p%R( o ), p%W( o ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%C_status( m ), p%X_status( n ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%o = o ; p%m = m ;              ! dimensions
   p%B = (/ 2.0_wp, 2.0_wp, 3.0_wp, 1.0_wp /) ! observations
   p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero
   p%W = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! weights
!  sparse row storage format
   CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', s ) ! Specify sparse row for Ao
   ALLOCATE( p%Ao%ptr( o + 1 ), p%Ao%col( ao_ne ), p%Ao%val( ao_ne ) )
   p%Ao%ptr = (/ 1, 3, 5, 7, 8 /) ! design matrix Ao
   p%Ao%col = (/ 1, 2, 2, 3, 1, 3, 2 /)
   p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem data complete
   CALL SLLSB_initialize( data, control, inform ) ! Initialize control params
   control%symmetric_linear_solver = 'sytr '
   control%FDC_control%symmetric_linear_solver = 'sytr '
   control%infinity = infinity                  ! Set infinity
   CALL SLLSB_solve( p, data, control, inform ) ! Solve
   IF ( inform%status == 0 ) THEN               ! Successful return
     WRITE( 6, "( ' SLLSB: ', I0, ' iterations  ', /,                          &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
   ELSE                                         !  Error returns
     WRITE( 6, "( ' SLLSB_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL SLLSB_terminate( data, control, inform )  !  delete internal workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%X_status )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
   END PROGRAM GALAHAD_SLLSB_EXAMPLE2
