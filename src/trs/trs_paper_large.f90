   PROGRAM GALAHAD_TRS_LARGE_EXAMPLE  !  GALAHAD 2.2 - 05/06/2008 AT 13:30 GMT.
   USE GALAHAD_TRS_DOUBLE                          ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )       ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
   INTEGER :: i, s, l, n2, logn, n, h_ne
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, X
   TYPE ( SMT_type ) :: H
   TYPE ( TRS_data_type ) :: data
   TYPE ( TRS_control_type ) :: control        
   TYPE ( TRS_inform_type ) :: inform
   REAL ( KIND = wp ) :: f = 0.0_wp           ! constant term, f
!  REAL ( KIND = wp ) :: radius = 0.001_wp    ! trust-region radius (small)
   REAL ( KIND = wp ) :: radius = 1.0_wp      ! trust-region radius (medium)
!  REAL ( KIND = wp ) :: radius = 100000.0_wp ! trust-region radius (large)
!$ INTEGER:: OMP_GET_MAX_THREADS
   CALL SMT_put( H%type, 'COORDINATE', s )    ! Specify co-ordinate for H
!$ WRITE(6, "( ' max threads = ', I0 )" ) OMP_GET_MAX_THREADS( )

!  DO logn = 6, 14
!  DO logn = 14, 14
!  DO logn = 13, 13
!  DO logn = 8, 8
   DO logn = 11, 11
!  DO logn = 17, 17
     IF ( MOD( logn, 2 ) == 1 ) THEN
       n = 3.1622772 * 10 ** ( logn / 2 )
       IF ( 2 * ( n / 2 ) /= n ) n = n + 1
     ELSE
       n = 10 ** ( logn / 2 )
     END IF
!    n = 10 ** logn
!    n = 2 ** logn
     h_ne = 4 * n - 6
     WRITE( 6, "( ' n, h_ne ', I0, 1X, I0 )" ) n, h_ne
     ALLOCATE( C( n ), X( n ) )
     ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) ) ; H%ne = h_ne
     n2 = n / 2
     DO l = 1, n
       C( l ) = - 0.5_wp
       H%row( l ) = l ;  H%col( l ) = l
       IF ( l /= 1 .AND. l /= n2 .AND. l /= n ) THEN
         H%val( l ) = 6.0_wp
       ELSE
         H%val( l ) = 2.0_wp * n + 10.0_wp
       END IF
     END DO

     l = n
     DO i = 2, n
       l = l + 1
       H%row( l ) = i ; H%col( l ) = 1
       IF ( i /= n2 .AND. i /= n ) THEN
         H%val( l ) = 2.0_wp
       ELSE
         H%val( l ) = 4.0_wp
       END IF
     END DO
     DO i = 2, n - 1
       l = l + 1
       H%row( l ) = n ; H%col( l ) = i
       IF ( i /= n2 ) THEN
         H%val( l ) = 2.0_wp
       ELSE
         H%val( l ) = 4.0_wp
       END IF
     END DO
     DO i = 2, n2 - 1
       l = l + 1
       H%row( l ) = n2 ; H%col( l ) = i ; H%val( l ) = 2.0_wp
     END DO     
     DO i = n2 + 1, n - 1
       l = l + 1
       H%row( l ) = i ; H%col( l ) = n2 ; H%val( l ) = 2.0_wp
     END DO     

     CALL TRS_initialize( data, control )       ! Initialize control parameters
     control%print_level = 3
!    control%equality_problem = .TRUE.
!    control%taylor_max_degree = 3
!    control%IR_control%itref_max = 2
     control%initial_multiplier = 0.0_wp
     control%use_initial_multiplier = .TRUE.
     control%SLS_control%ordering = 2
!    control%problem = 99
     control%definite_linear_solver = 'ma87'
     control%SLS_control%initial_pool_size = 6400000

write(6,*) ' ** solver ', control%definite_linear_solver
     CALL TRS_solve( n, radius, f, C, H, X, data, control, inform )
     IF ( inform%status == 0 ) THEN !  Successful return
       WRITE( 6, "( ' Solution and Lagrange multiplier =', 2ES12.4 )" )         &
         inform%obj, inform%multiplier
       WRITE( 6, "( 1X, I0,' factorizations, time = ', F9.2 )" )                &
         inform%factorizations, inform%time%total
     ELSE  !  Error returns
       WRITE( 6, "( ' TRS_solve exit status = ', I0 ) " ) inform%status
     END IF
     CALL TRS_terminate( data, control, inform )  ! delete internal workspace
     DEALLOCATE( X, C, H%row, H%col, H%val )

write(6,*) ' factors ', inform%SLS_inform%entries_in_factors
   END DO
   END PROGRAM GALAHAD_TRS_LARGE_EXAMPLE
