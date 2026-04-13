! THIS VERSION: GALAHAD 5.5 - 2026-02-01 AT 08:40 GMT
   PROGRAM GALAHAD_SLLS_MULTIPLE_EXAMPLE
   USE GALAHAD_SLLS_double ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( USERDATA_type ) :: userdata
   INTEGER :: i, j, m, s
   INTEGER, PARAMETER :: n = 12, o = n + 1, a_ne = 2 * n + 2
! set problem data
   ALLOCATE( p%B( o ), p%X( n ), p%X_status( n ), p%COHORT( n ) )
   DO j = 1, n  ! cohorts
     SELECT CASE( MOD( j, 3 ) )
     CASE ( 1 )
       p%COHORT( j ) = 1
     CASE ( 2 )
       p%COHORT( j ) = 2
     CASE DEFAULT
      p%COHORT( j ) = 0
     END SELECT
   END DO
   m = MAXVAL( p%COHORT ) ! number of cohorts
   p%B( 1 ) = 0.0_wp  ! observations
   DO i = 2, o
     SELECT CASE( MOD( i, 3 ) )
     CASE ( 1 )
       p%B( i ) = 2.0_wp
     CASE DEFAULT
       p%B( i ) = 1.0_wp
     END SELECT
   END DO
   p%n = n ; p%o = o ; p%m = m ! dimensions
   p%X = 0.0_wp ! start from zero
!  sparse co-ordinate storage format for the design matrix Ao
   CALL SMT_put( p%Ao%type, 'COORDINATE', s ) ! Co-ordinate  storage for Ao
   ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( A_ne ) )
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
   p%regularization_weight = 0.1_wp
   ALLOCATE( p%X_s( n ) )
   p%X_s( n ) = 1.0_wp
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 3 ! print one line/iteration
   control%maxit = 20      ! maximum of 20 iterations
!  control%maxit = 2       ! maximum of 2 iterations
   control%SBLS_control%symmetric_linear_solver = 'sytr '
   control%SBLS_control%definite_linear_solver = 'potr '
!  control%direct_subproblem_solve = .FALSE.
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN  !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution =', /, ( 5ES12.4 ) )" )              &
       inform%iter, inform%obj, p%X
     WRITE( 6, "( ' Lagrange multiplier estimates =', ( 5ES12.4 ) )" ) p%Y
   ELSE                            ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, "( /, ' objective value =',                                     &
    &       ES12.4, /, ' Current solution =', ( 5ES12.4 ) )" ) inform%obj, p%X
   END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%G, p%X_status, p%COHORT )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
   END PROGRAM GALAHAD_SLLS_MULTIPLE_EXAMPLE
