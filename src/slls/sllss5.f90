! THIS VERSION: GALAHAD 5.5 - 2026-02-01 AT 08:40 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE5
   USE GALAHAD_SLLS_double                   ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( USERDATA_type ) :: userdata
   INTEGER :: n, o, m, ao_ne, i, s
   REAL :: t1, t2
   REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( 1.0_wp )
   LOGICAL, PARAMETER :: old_data = .FALSE.

! start problem data

   CALL CPU_TIME( t1 )

!  read data from file

   IF ( old_data ) THEN
     OPEN( 31, file = "old_sllss5.data" )
     READ( 31, * ) n, o, m, ao_ne
     ALLOCATE( p%B( o ), p%X( n ), p%COHORT( n ), p%X_status( n ), p%X_s( n ) )
     ALLOCATE( p%Ao%val( ao_ne ) )
     CALL SMT_put( p%Ao%type, 'DENSE', s )
     p%n = n ; p%o = o ; p%m = m ; p%Ao%n = n ; p%Ao%m = o ; p%Ao%ne = ao_ne
     READ( 31, * ) p%Ao%val
     READ( 31, * ) p%B
     READ( 31, * ) p%COHORT
     READ( 31, * ) p%regularization_weight
     READ( 31, * ) p%X
     READ( 31, * ) p%X_s
     CLOSE( 31 )
     DO i = 1, n
       IF ( ABS( p%X( i ) ) <= epsmch ) p%X( i ) = 0.0_wp
       IF ( ABS( p%X_s( i ) ) <= epsmch ) p%X_s( i ) = 0.0_wp
     END DO
     OPEN( 32 )
     WRITE( 32, * ) n, o, m, ao_ne
     WRITE( 32, * ) p%regularization_weight
     DO i = 1, ao_ne
       WRITE( 32, * ) p%Ao%val( i )
     END DO
     DO i = 1, o
       WRITE( 32, * ) p%B( i )
     END DO
     DO i = 1, n
       WRITE( 32, * ) p%COHORT( i ), p%X( i ), p%X_s( i )
     END DO
     CLOSE( 32 )
   ELSE
     OPEN( 32, file = "sllss5.data" )
     READ( 32, * ) n, o, m, ao_ne
     ALLOCATE( p%B( o ), p%X( n ), p%COHORT( n ), p%X_status( n ), p%X_s( n ) )
     ALLOCATE( p%Ao%val( ao_ne ) )
     CALL SMT_put( p%Ao%type, 'DENSE', s )
     p%n = n ; p%o = o ; p%m = m ; p%Ao%n = n ; p%Ao%m = o ; p%Ao%ne = ao_ne
     READ( 32, * ) p%regularization_weight
     DO i = 1, ao_ne
       READ( 32, * ) p%Ao%val( i )
     END DO
     DO i = 1, o
       READ( 32, * ) p%B( i )
     END DO
     DO i = 1, n
       READ( 32, * ) p%COHORT( i ), p%X( i ), p%X_s( i )
     END DO
     CLOSE( 32 )
   END IF
   CALL CPU_TIME( t2 )
   WRITE( 6, "( ' CPU time for data read = ', F6.2 )" ) t2 - t1
!  stop
   WRITE( 6, "( ' min, max A, b = ', 3ES12.4 )" )                              &
     MINVAL( ABS( p%Ao%val ) ), MAXVAL( ABS( p%Ao%val ) ), MAXVAL( ABS( p%B ) )

!  sparse co-ordinate storage format

! problem data complete
   CALL SLLS_initialize( data, control, inform ) ! Initialize control parameters
   control%SBLS_control%symmetric_linear_solver = 'sytr' ! non-default solver
   control%SBLS_control%definite_linear_solver = 'potr' ! non-default solver
!  control%print_level = 1                       ! print one line/iteration
!  control%out = 66
   control%print_level = 3                       ! print multiple line/iteration
!  control%start_print = 379
!  control%maxit = 2
!  control%direct_subproblem_solve = .FALSE.
!  control%exact_arc_search = .FALSE.
   control%stop_d = ( 10.0_wp ) ** ( - 9 )

!p%regularization_weight = 0.0_wp
!p%COHORT(1:2)=2
!p%m = 2
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN             !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =', ES12.4 )" ) inform%iter, inform%obj
!    WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p%X
!    WRITE( 6, "( ' Lagrange multiplier estimate =', ES12.4 )" ) p%Y
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
   END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace
   DEALLOCATE( p%B, p%X, p%Y, p%Z, p%R, p%G, p%X_status, p%X_s )
   DEALLOCATE( p%Ao%val, p%Ao%type )
   END PROGRAM GALAHAD_SLLS_EXAMPLE5
