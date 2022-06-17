! THIS VERSION: GALAHAD 4.1 - 2022-06-16 AT 09:45 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE7

! use slls to solve a multilinear fitting problem
!  min 1/2 sum_i ( xi sum_il sum_ir sum_it sum_ip a(il,ir,it,ip,i)
!                  l_il r_ir theta_it y_phi + b - mu_i)^2,
! where u, v, w and x lie in n and m regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with u free and v,x,y fixed, and vice versa

   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_NORMS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
   REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 20
   TYPE ( QPT_problem_type ) :: p_l, p_r, p_t, p_p
   TYPE ( SLLS_data_type ) :: data_l, data_r, data_t, data_p
   TYPE ( SLLS_control_type ) :: control_l, control_r, control_t, control_p
   TYPE ( SLLS_inform_type ) :: inform_l, inform_r, inform_t, inform_p
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: i, il, ir, it, ip, ix, iy, j, ne, pass
   REAL ( KIND = wp ) :: val, f, xi, b, a_11, a_12, a_22, r_1, r_2, f_best
   REAL ( KIND = wp ) :: alpha, beta, prod1, prod2, prod3, prod4
   REAL :: time, time_start, time_total
   INTEGER, PARAMETER :: nl = 18
   INTEGER, PARAMETER :: nr = 17
   INTEGER, PARAMETER :: nt = 16
   INTEGER, PARAMETER :: np = 15
   INTEGER, PARAMETER :: qx = 60
   INTEGER, PARAMETER :: qy = 60
   INTEGER, PARAMETER :: obs = qx * qy
   INTEGER, PARAMETER :: pass_max = 1000
   LOGICAL, PARAMETER :: prints = .FALSE.
!  LOGICAL, PARAMETER :: solution = .TRUE.
   LOGICAL, PARAMETER :: solution = .FALSE.
   REAL ( KIND = wp ), PARAMETER :: f_stop = ten ** ( - 16 )
   REAL ( KIND = wp ), PARAMETER :: scale = ten ** 15
   LOGICAL, PARAMETER :: pertub_observations = .TRUE.
!  LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   INTEGER :: L_stat( nl ), R_stat( nr ), T_stat( nt ), P_stat( np )
   REAL ( KIND = wp ) :: L( nl ), R( nr ), THETA( nt ), PHI( np )
   REAL ( KIND = wp ) :: MU( obs ), SIGMA( obs ), RHS( qx, qy )
   REAL ( KIND = wp ) :: A( nl, nr, nt, np, obs ), A_SAVE( obs )

!  start problem data

   CALL CPU_TIME( time_start )

!  starting point
   IF ( solution ) THEN
     OPEN( unit = 23, file = "/numerical/matrices/sas/cylinder/l.txt" )
     DO j = 1, nl
       READ( 23, * ) L( j )
     END DO
     CLOSE( unit = 24 )
     OPEN( unit = 24, file = "/numerical/matrices/sas/cylinder/r.txt" )
     DO j = 1, nr
       READ( 24, * ) R( j )
     END DO
     CLOSE( unit = 24 )
     OPEN( unit = 25, file = "/numerical/matrices/sas/cylinder/theta.txt" )
     DO j = 1, nt
       READ( 25, * ) THETA( j )
     END DO
     CLOSE( unit = 25 )
     OPEN( unit = 26, file = "/numerical/matrices/sas/cylinder/phi.txt" )
     DO j = 1, np
       READ( 26, * ) PHI( j )
     END DO
     CLOSE( unit = 26 )
     xi = 2.351270579774912875d+3 ; b = 2.2d-4
   ELSE
     L = 1.0_wp / REAL( nl, KIND = wp ) ; R = 1.0_wp / REAL( nr, KIND = wp )
     THETA = 1.0_wp / REAL( nt, KIND = wp ) ; PHI = 1.0_wp / REAL( np, KIND=wp )
     xi = 3.0_wp ; b = 1.0_wp
   END IF

!  input observations mu (and set sigma to be mu)

   OPEN( unit = 21, file = "/numerical/matrices/sas/cylinder/intensities.txt" )
   ne = 0
   DO j = 1, qy
!    READ( 21, * ) ( MU( i ), i = ne + 1, ne + qx )
     READ( 21, * ) RHS( j, 1 : qx )
     ne = ne + qx
   END DO
   CLOSE( unit = 21 )

   ne = 0
   DO j = 1, qx
     DO i = 1, qy
       ne = ne + 1
       MU( ne ) = RHS( i, j )
    END DO
  END DO

!  scale mu

   SIGMA( 1 : obs ) = MU( 1 : obs )
   MU( 1 : obs ) = 1.0_wp

!  read problem data. Input tensor A, and scale

   OPEN( unit = 22, file = "/numerical/matrices/sas/cylinder/G.txt" )
   j = 0
   DO
     READ( 22, *, END = 1 ) ix, iy, il, ir, it, ip, val
     i = qx * iy + ix + 1
     A( il + 1, ir + 1, it + 1, ip + 1, i ) = val / ( scale * SIGMA( i ) )
     j = j + 1
     IF ( MOD( j, 10000000 ) == 0 ) THEN
       CALL CPU_TIME( time ) ; time_total = time - time_start
       WRITE( 6, "( ' line ', I0, ' read, CPU time so far = ', F0.2 )" )       &
        j, time_total
     END IF
   END DO
 1 CONTINUE
   CLOSE( unit = 23 )

write(6,*) ' max A = ', MAXVAL( ABS( A ) )

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total
   WRITE( 6, "( 1X, I0, ' lines read' )" ) j
!   stop

!  set up storage for the u problem

   CALL SMT_put( p_l%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_l%A%val(obs * nl ), p_l%B( obs ), p_l%X( nl ) )
   p_l%m = obs ; p_l%n = nl ; p_l%A%m = p_l%m ; p_l%A%n = p_l%n

!  set up storage for the v problem

   CALL SMT_put( p_r%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_r%A%val(obs * nr ), p_r%B( obs ), p_r%X( nr ) )
   p_r%m = obs ; p_r%n = nr ; p_r%A%m = p_r%m ; p_r%A%n = p_r%n

!  set up storage for the x problem

   CALL SMT_put( p_t%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_t%A%val(obs * nt ), p_t%B( obs ), p_t%X( nt ) )
   p_t%m = obs ; p_t%n = nt ; p_t%A%m = p_t%m ; p_t%A%n = p_t%n

!  set up storage for the y problem

   CALL SMT_put( p_p%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_p%A%val(obs * np ), p_p%B( obs ), p_p%X( np ) )
   p_p%m = obs ; p_p%n = np ; p_p%A%m = p_p%m ; p_p%A%n = p_p%n

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_l, control_l, inform_l )
   control_l%infinity = infinity
   IF ( prints ) THEN
     control_l%print_level = 1
   ELSE
     control_l%print_level = 0
   END IF
   control_l%exact_arc_search = .FALSE.
!  control_l%convert_control%print_level = 3

   CALL SLLS_initialize( data_r, control_r, inform_r )
   control_r%infinity = infinity
   IF ( prints ) THEN
     control_r%print_level = 1
   ELSE
     control_r%print_level = 0
   END IF
   control_r%exact_arc_search = .FALSE.
!  control_r%convert_control%print_level = 3

   CALL SLLS_initialize( data_t, control_t, inform_t )
   control_t%infinity = infinity
   IF ( prints ) THEN
     control_t%print_level = 1
   ELSE
     control_t%print_level = 0
   END IF
   control_t%exact_arc_search = .FALSE.
!  control_t%convert_control%print_level = 3

   CALL SLLS_initialize( data_p, control_p, inform_p )
   control_p%infinity = infinity
   IF ( prints ) THEN
     control_p%print_level = 1
   ELSE
     control_p%print_level = 0
   END IF
   control_p%exact_arc_search = .FALSE.
!  control_p%convert_control%print_level = 3

   CALL CPU_TIME( time_start )

!  fix u, v, x, y, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     alpha = 0.0_wp
     DO ip = 1, np
       prod1 = PHI( ip )
       DO it = 1, nt
         prod2 = prod1 * THETA( it )
         DO ir = 1, nr
           prod3 = prod2 * R( ir )
           DO il = 1, nl
             prod4 = prod3 * L( il )
             alpha = alpha + A( il, ir, it, ip, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
!write(6,"( ' rhs ', I0, ' = ', ES12.4 )" ) i,  (alpha * xi + beta * b ) * SIGMA(i)
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current mu, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved mu, b = ', 2ES22.14 )" ) xi, b

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total

!  pass

   pass = 1
10 continue
   f_best = f

   WRITE( 6, "( /, 1X, 32( '=' ), ' pass ', I0, 1X, 32( '=' ) )" ) pass

!  fix r, theta, phi, xi and b, and solve for l

   p_l%X = L

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO il = 1, nl
       val = 0.0_wp
       DO ip = 1, np
         prod1 = xi * PHI( ip )
         DO it = 1, nt
           prod2 = prod1 * THETA( it )
           DO ir = 1, nr
             val = val + A( il, ir, it, ip, i ) * prod2 * R( ir )
           END DO
         END DO
       END DO
       p_l%A%val( ne + il ) = val
     END DO
     ne = ne + nl
     p_l%B( i ) = MU( i ) - b / SIGMA( i )
   END DO

!  solve the l problem

   inform_l%status = 1
   CALL SLLS_solve( p_l, L_stat, data_l, control_l, inform_l, userdata )
   IF ( inform_l%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new u', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_l%iter, inform_l%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_l%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_l%status
     WRITE( 6, * ) inform_l%alloc_status, inform_l%bad_alloc
   END IF

!  record the new l

   L = p_l%X

!  fix l, theta, phi, xi and b, and solve for r

   p_r%X = R

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO ir = 1, nr
       val = 0.0_wp
       DO ip = 1, np
         prod1 = xi * PHI( ip )
         DO it = 1, nt
           prod2 = prod1 * THETA( it )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_r%A%val( ne + ir ) = val
     END DO
     ne = ne + nr
     p_r%B( i ) = MU( i ) - b / SIGMA( i )
   END DO

!  solve the v problem

   inform_r%status = 1
   CALL SLLS_solve( p_r, R_stat, data_r, control_r, inform_r, userdata )
   IF ( inform_r%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new v', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_r%iter, inform_r%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_r%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_r%status
     WRITE( 6, * ) inform_r%alloc_status, inform_r%bad_alloc
   END IF

!  record the new r

   R = p_r%X

!  fix l, r, phi, xi and b, and solve for theta

   p_t%X = THETA

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO it = 1, nt
       val = 0.0_wp
       DO ip = 1, np
         prod1 = xi * PHI( ip )
         DO ir = 1, nr
           prod2 = prod1 * R( ir )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_t%A%val( ne + it ) = val
     END DO
     ne = ne + nt
     p_t%B( i ) = MU( i ) - b / SIGMA( i )
   END DO

!  solve the theta problem

   inform_t%status = 1
   CALL SLLS_solve( p_t, T_stat, data_t, control_t, inform_t, userdata )
   IF ( inform_t%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new x', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_t%iter, inform_t%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_t%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_t%status
     WRITE( 6, * ) inform_t%alloc_status, inform_t%bad_alloc
   END IF

!  record the new theta

   THETA = p_t%X

!  fix l, r, theta, xi and b, and solve for phi

   p_p%X = PHI

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO ip = 1, np
       val = 0.0_wp
       DO it = 1, nt
         prod1 = xi * THETA( it )
         DO ir = 1, nr
           prod2 = prod1 * R( ir )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_p%A%val( ne + ip ) = val
     END DO
     ne = ne + np
     p_p%B( i ) = MU( i ) - b / SIGMA( i )
   END DO

!  solve the phi problem

   inform_p%status = 1
   CALL SLLS_solve( p_p, P_stat, data_p, control_p, inform_p, userdata )
   IF ( inform_p%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new y', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_p%iter, inform_p%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_p%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_p%status
     WRITE( 6, * ) inform_p%alloc_status, inform_p%bad_alloc
   END IF

!  record the new phi

   PHI = p_p%X

!  fix u, v, x, y, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     alpha = 0.0_wp
     DO ip = 1, np
       prod1 = PHI( ip )
       DO it = 1, nt
         prod2 = prod1 * THETA( it )
         DO ir = 1, nr
           prod3 = prod2 * R( ir )
           DO il = 1, nl
             prod4 = prod3 * L( il )
             alpha = alpha + A( il, ir, it, ip, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + b / SIGMA( i )- MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) f
   WRITE( 6, "( ' current mu, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved mu, b = ', 2ES22.14 )" ) xi / scale, b

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total

   pass = pass + 1
   IF ( pass <= pass_max .AND. f < f_best .AND. f > f_stop ) GO TO 10

!  end of loop

!  report final residuals

   f = 0.0_wp
   DO i = 1, obs
     alpha = 0.0_wp
     DO ip = 1, np
       prod1 = PHI( ip )
       DO it = 1, nt
         prod2 = prod1 * THETA( it )
         DO ir = 1, nr
           prod3 = prod2 * R( ir )
           DO il = 1, nl
             prod4 = prod3 * L( il )
             alpha = alpha + A( il, ir, it, ip, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     f = f + ( alpha * xi + b / SIGMA( i )- MU( i ) ) ** 2
   END DO

   WRITE( 6, "( /, 1X, 29( '=' ), ' pass limit ', 1X, 29( '=' ) )" )
   WRITE( 6, "( ' Optimal l = ', /, ( 5ES12.4 ) )" ) L
   WRITE( 6, "( ' Optimal r = ', /, ( 5ES12.4 ) )" ) R
   WRITE( 6, "( ' Optimal theta = ', /, ( 5ES12.4 ) )" ) THETA
   WRITE( 6, "( ' Optimal phi = ', /, ( 5ES12.4 ) )" ) PHI
   WRITE( 6, "( ' Optimal xi, b = ', /, 2ES22.14 )" ) xi / scale, b
   WRITE( 6, "( /, ' Optimal value =', ES22.14 )" ) 0.5_wp * f
   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( ' total CPU time = ', F0.2, ' passes = ', I0 )" )              &
     time_total, pass

! tidy up afterwards

   CALL SLLS_terminate( data_l, control_l, inform_l )  !  delete workspace
   DEALLOCATE( p_l%B, p_l%X, p_l%Z )
   DEALLOCATE( p_l%A%val, p_l%A%type )

   CALL SLLS_terminate( data_r, control_r, inform_r )  !  delete workspace
   DEALLOCATE( p_r%B, p_r%X, p_r%Z )
   DEALLOCATE( p_r%A%val, p_r%A%type )

   CALL SLLS_terminate( data_t, control_t, inform_t )  !  delete workspace
   DEALLOCATE( p_t%B, p_t%X, p_t%Z )
   DEALLOCATE( p_t%A%val, p_t%A%type )

   CALL SLLS_terminate( data_p, control_p, inform_p )  !  delete workspace
   DEALLOCATE( p_p%B, p_p%X, p_p%Z )
   DEALLOCATE( p_p%A%val, p_p%A%type )

   END PROGRAM GALAHAD_SLLS_EXAMPLE7
