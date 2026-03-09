! THIS VERSION: GALAHAD 5.5 - 2026-02-19 AT 10:10 GMT.
   PROGRAM GALAHAD_SLLS_SECOND_EXAMPLE ! reverse commmunication interface
   USE GALAHAD_SLLS_double             ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( REVERSE_type ) :: reverse
   TYPE ( USERDATA_type ) :: userdata
   INTEGER :: i, j, k, l
   REAL ( KIND = wp ) :: val
   INTEGER, PARAMETER :: n = 3, o = 4, m = 1, Ao_ne = 5
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_ptr
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ao_val
! start problem data
   ALLOCATE( p%B( o ), p%X( n ), p%X_status( n ) )
   p%n = n ; p%o = o ; p%m = m                ! dimensions
   p%B = (/ 0.0_wp, 2.0_wp, 1.0_wp, 2.0_wp /) ! right-hand side
   p%X = 0.0_wp ! start from zero
!  sparse column storage format
   ALLOCATE( Ao_val( Ao_ne ), Ao_row( Ao_ne ), Ao_ptr( n + 1 ) )
   Ao_row = (/ 1, 2, 2, 3, 4 /) ! ! design matrix Ao by columns: row indices
   Ao_ptr = (/ 1, 3, 4, 6 /)      ! pointers to column starts
   Ao_val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) 
! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
!  control%print_level = 1                       ! print one line/iteration
   control%exact_arc_search = .FALSE.
   inform%status = 1
10 CONTINUE ! Solve problem - reverse commmunication loop
     CALL SLLS_solve( p, data, control, inform, userdata, reverse = reverse )
     SELECT CASE ( inform%status )
     CASE ( 0 ) !  successful return
       WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                      &
      &     ' Optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
         inform%iter, inform%obj, p%X
       WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
     CASE ( 2 ) ! compute Ao * v
       reverse%P( : o ) = 0.0_wp
       DO j = 1, n
         val = reverse%V( j )
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           i = Ao_row( k )
           reverse%P( i ) = reverse%P( i ) + Ao_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 3 ) ! compute Ao^T * v
       reverse%P( : n ) = 0.0_wp
       DO j = 1, n
         val = 0.0_wp
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           val = val + Ao_val( k ) * reverse%V( Ao_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE ( 4 ) ! compute a column of Ao
       reverse%lp = 0
       DO k = Ao_ptr( reverse%index ), Ao_ptr( reverse%index + 1 ) - 1
         reverse%lp = reverse%lp + 1
         reverse%P( reverse%lp ) = Ao_val( k )
         reverse%IP( reverse%lp ) = Ao_row( k )
       END DO
       GO TO 10
     CASE ( 5 ) ! compute Ao * sparse v
       reverse%P( : o ) = 0.0_wp
       DO l = reverse%lvl, reverse%lvu
         j = reverse%IV( l )
         val = reverse%V( j )
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           i = Ao_row( k )
           reverse%P( i ) = reverse%P( i ) + Ao_val( k ) * val
         END DO
       END DO
       GO TO 10
     CASE ( 6 ) ! compute sparse( Ao^T * v )
       reverse%P( : n ) = 0.0_wp
       DO l = reverse%lvl, reverse%lvu
         j = reverse%IV( l )
         val = 0.0_wp
         DO k = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
           val = val + Ao_val( k ) * reverse%V( Ao_row( k ) )
         END DO
         reverse%P( j ) = val
       END DO
       GO TO 10
     CASE DEFAULT ! error returns
       WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     END SELECT
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%X_status )
   DEALLOCATE( Ao_val, Ao_row, Ao_ptr )
   END PROGRAM GALAHAD_SLLS_SECOND_EXAMPLE
