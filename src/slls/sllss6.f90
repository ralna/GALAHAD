! THIS VERSION: GALAHAD 4.3 - 2023-12-31 AT 10:15 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE6

! use slls to solve a multilinear fitting problem
!  min 1/2 sum_i ( xi sum_iu sum_iv sum_ix sum_iy a(i,iu,iv,ix,iy)
!                  u_iu v_iv x_ix y_iy + b - mu_i)^2,
! where u, v, w and x lie in n and o regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with u free and v,x,y fixed, and vice versa

   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_RAND_double
   USE GALAHAD_NORMS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p_u, p_v, p_x, p_y
   TYPE ( SLLS_data_type ) :: data_u, data_v, data_x, data_y
   TYPE ( SLLS_control_type ) :: control_u, control_v, control_x, control_y
   TYPE ( SLLS_inform_type ) :: inform_u, inform_v, inform_x, inform_y
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: i, iu, iv, ix, iy, ne, pass
   TYPE ( RAND_seed ) :: seed
   REAL ( KIND = wp ) :: val, f, xi, b, a_11, a_12, a_22, r_1, r_2, f_best
   REAL ( KIND = wp ) :: prod, prod1, prod2, prod3, prod4
   REAL :: time, time_start, time_total
!  INTEGER, PARAMETER :: nu = 4
   INTEGER, PARAMETER :: nu = 10
!  INTEGER, PARAMETER :: nv = 8
   INTEGER, PARAMETER :: nv = 10
!  INTEGER, PARAMETER :: nx = 12
   INTEGER, PARAMETER :: nx = 10
!  INTEGER, PARAMETER :: ny = 16
   INTEGER, PARAMETER :: ny = 10
!  INTEGER, PARAMETER :: obs = 80
   INTEGER, PARAMETER :: obs = 10000
   INTEGER, PARAMETER :: pass_max = 100
   LOGICAL, PARAMETER :: prints = .FALSE.
   REAL ( KIND = wp ), PARAMETER :: f_stop = ( 10.0_wp ) ** ( - 16 )
   LOGICAL, PARAMETER :: pertub_observations = .TRUE.
!  LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   INTEGER :: U_stat( nu ), V_stat( nv ), X_stat( nx ), Y_stat( ny )
   REAL ( KIND = wp ) :: U( nu ), V( nv ), X( nx ), Y( ny ), MU( obs )
   REAL ( KIND = wp ) :: A( nu, nv, nx, ny, obs ), ALPHA( obs )

! start problem data

! generate sparse solution

   CALL RAND_initialize( seed )
   CALL RAND_random_real( seed, .FALSE., U( : nu ) )
   U = MAX( U, 0.0_wp )
   val = SUM( U( : nu ) ) ; U( : nu ) = U( : nu ) / val
   CALL RAND_random_real( seed, .FALSE., V( : nv ) )
   V = MAX( V, 0.0_wp )
   val = SUM( V( : nv ) ) ; V( : nv ) = V( : nv ) / val
   CALL RAND_random_real( seed, .FALSE., X( : nx ) )
   X = MAX( X, 0.0_wp )
   val = SUM( X( : nx ) ) ; X( : nx ) = X( : nx ) / val
   CALL RAND_random_real( seed, .FALSE., Y( : ny ) )
   Y = MAX( Y, 0.0_wp )
   val = SUM( Y( : ny ) ) ; Y( : ny ) = Y( : ny ) / val

   xi = 2.0_wp ; b = 0.5_wp

!  random A

   DO i = 1, obs
     DO iy = 1, ny
       DO ix = 1, nx
         DO iv = 1, nv
           DO iu = 1, nu
             CALL RAND_random_real( seed, .TRUE., A( iu, iv, ix, iy, i ) )
           END DO
         END DO
       END DO
     END DO
   END DO

!  generate a set of observations, mu_i

   DO i = 1, obs
     IF ( pertub_observations ) THEN
       CALL RAND_random_real( seed, .TRUE., val ) ; val = 0.001 * val
     ELSE
       val = 0.0_wp
     END IF
     prod = xi
     DO iy = 1, ny
       prod1 = prod * Y( iy )
       DO ix = 1, nx
         prod2 = prod1 * X( ix )
         DO iv = 1, nv
           prod3 = prod2 * V( iv )
           DO iu = 1, nu
             prod4 = prod3 * U( iu )
             val = val + A( iu, iv, ix, iy, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     MU( i ) = val + b
   END DO

!  set up storage for the u problem

   CALL SMT_put( p_u%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_u%Ao%val(obs * nu ), p_u%B( obs ), p_u%X( nu ) )
   p_u%o = obs ; p_u%n = nu ; p_u%Ao%m = p_u%o ; p_u%Ao%n = p_u%n

!  set up storage for the v problem

   CALL SMT_put( p_v%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_v%Ao%val(obs * nv ), p_v%B( obs ), p_v%X( nv ) )
   p_v%o = obs ; p_v%n = nv ; p_v%Ao%m = p_v%o ; p_v%Ao%n = p_v%n

!  set up storage for the x problem

   CALL SMT_put( p_x%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_x%Ao%val(obs * nx ), p_x%B( obs ), p_x%X( nx ) )
   p_x%o = obs ; p_x%n = nx ; p_x%Ao%m = p_x%o ; p_x%Ao%n = p_x%n

!  set up storage for the y problem

   CALL SMT_put( p_y%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_y%Ao%val(obs * ny ), p_y%B( obs ), p_y%X( ny ) )
   p_y%o = obs ; p_y%n = ny ; p_y%Ao%m = p_y%o ; p_y%Ao%n = p_y%n

!  starting point
   U = 1.0_wp / REAL( nu, KIND = wp ) ; V = 1.0_wp / REAL( nv, KIND = wp )
   X = 1.0_wp / REAL( nx, KIND = wp ) ; Y = 1.0_wp / REAL( ny, KIND = wp )
   xi = 3.0_wp ; b = 1.0_wp

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_u, control_u, inform_u )
   IF ( prints ) THEN
     control_u%print_level = 1
   ELSE
     control_u%print_level = 0
   END IF
   control_u%exact_arc_search = .FALSE.
!  control_u%convert_control%print_level = 3

   CALL SLLS_initialize( data_v, control_v, inform_v )
   IF ( prints ) THEN
     control_v%print_level = 1
   ELSE
     control_v%print_level = 0
   END IF
   control_v%exact_arc_search = .FALSE.
!  control_v%convert_control%print_level = 3

   CALL SLLS_initialize( data_x, control_x, inform_x )
   IF ( prints ) THEN
     control_x%print_level = 1
   ELSE
     control_x%print_level = 0
   END IF
   control_x%exact_arc_search = .FALSE.
!  control_x%convert_control%print_level = 3

   CALL SLLS_initialize( data_y, control_y, inform_y )
   IF ( prints ) THEN
     control_y%print_level = 1
   ELSE
     control_y%print_level = 0
   END IF
   control_y%exact_arc_search = .FALSE.
!  control_y%convert_control%print_level = 3

   CALL CPU_TIME( time_start )

!  fix u, v, x, y, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     val = 0.0_wp
     DO iy = 1, ny
       prod1 = Y( iy )
       DO ix = 1, nx
         prod2 = prod1 * X( ix )
         DO iv = 1, nv
           prod3 = prod2 * V( iv )
           DO iu = 1, nu
             prod4 = prod3 * U( iu )
             val = val + A( iu, iv, ix, iy, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     ALPHA( i ) = val
     f = f + ( val * xi + b - MU( i ) ) ** 2
     a_11 = a_11 + val * val
     a_12 = a_12 + val
     a_22 = a_22 + 1.0_wp
     r_1 = r_1 + val * MU( i )
     r_2 = r_2 + MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current mu, b = ', 2ES22.14 )" ) xi, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( ALPHA( i ) * xi + b - MU( i ) ) ** 2
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

!  fix v, x, y, xi and b, and solve for u

   p_u%X = U

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO iu = 1, nu
       val = 0.0_wp
       DO iy = 1, ny
         prod1 = xi * Y( iy )
         DO ix = 1, nx
           prod2 = prod1 * X( ix )
           DO iv = 1, nv
             val = val + A( iu, iv, ix, iy, i ) * prod2 * V( iv )
           END DO
         END DO
       END DO
       p_u%Ao%val( ne + iu ) = val
     END DO
     ne = ne + nu
     p_u%B( i ) = MU( i ) - b
   END DO

!  solve the u problem

   inform_u%status = 1
   CALL SLLS_solve( p_u, U_stat, data_u, control_u, inform_u, userdata )
   IF ( inform_u%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new u', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_u%iter, inform_u%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_u%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_u%status
     WRITE( 6, * ) inform_u%alloc_status, inform_u%bad_alloc
   END IF
   CALL CPU_TIME( time ) ; time_total = time - time_start

!  record the new u

   U = p_u%X

!  fix u, x, y, xi and b, and solve for v

   p_v%X = V

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO iv = 1, nv
       val = 0.0_wp
       DO iy = 1, ny
         prod1 = xi * Y( iy )
         DO ix = 1, nx
           prod2 = prod1 * X( ix )
           DO iu = 1, nu
             val = val + A( iu, iv, ix, iy, i ) * prod2 * U( iu )
           END DO
         END DO
       END DO
       p_v%Ao%val( ne + iv ) = val
     END DO
     ne = ne + nv
     p_v%B( i ) = MU( i ) - b
   END DO

!  solve the v problem

   inform_v%status = 1
   CALL SLLS_solve( p_v, V_stat, data_v, control_v, inform_v, userdata )
   IF ( inform_v%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new v', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_v%iter, inform_v%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_v%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_v%status
     WRITE( 6, * ) inform_v%alloc_status, inform_v%bad_alloc
   END IF

!  record the new x

   V = p_v%X

!  fix u, v, y, xi and b, and solve for x

   p_x%X = X

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO ix = 1, nx
       val = 0.0_wp
       DO iy = 1, ny
         prod1 = xi * Y( iy )
         DO iv = 1, nv
           prod2 = prod1 * V( iv )
           DO iu = 1, nu
             val = val + A( iu, iv, ix, iy, i ) * prod2 * U( iu )
           END DO
         END DO
       END DO
       p_x%Ao%val( ne + ix ) = val
     END DO
     ne = ne + nx
     p_x%B( i ) = MU( i ) - b
   END DO

!  solve the v problem

   inform_x%status = 1
   CALL SLLS_solve( p_x, X_stat, data_x, control_x, inform_x, userdata )
   IF ( inform_x%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new x', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_x%iter, inform_x%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_x%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_x%status
     WRITE( 6, * ) inform_x%alloc_status, inform_x%bad_alloc
   END IF

!  record the new x

   X = p_x%X

!  fix u, v, x, xi and b, and solve for y

   p_y%X = Y

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO iy = 1, ny
       val = 0.0_wp
       DO ix = 1, nx
         prod1 = xi * X( ix )
         DO iv = 1, nv
           prod2 = prod1 * V( iv )
           DO iu = 1, nu
             val = val + A( iu, iv, ix, iy, i ) * prod2 * U( iu )
           END DO
         END DO
       END DO
       p_y%Ao%val( ne + iy ) = val
     END DO
     ne = ne + ny
     p_y%B( i ) = MU( i ) - b
   END DO

!  solve the v problem

   inform_y%status = 1
   CALL SLLS_solve( p_y, Y_stat, data_y, control_y, inform_y, userdata )
   IF ( inform_y%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new y', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_y%iter, inform_y%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_y%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_y%status
     WRITE( 6, * ) inform_y%alloc_status, inform_y%bad_alloc
   END IF

!  record the new x

   Y = p_y%X

!  fix u, v, x, y, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     val = 0.0_wp
     DO iy = 1, ny
       prod1 = Y( iy )
       DO ix = 1, nx
         prod2 = prod1 * X( ix )
         DO iv = 1, nv
           prod3 = prod2 * V( iv )
           DO iu = 1, nu
             prod4 = prod3 * U( iu )
             val = val + A( iu, iv, ix, iy, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     ALPHA( i ) = val
     f = f + ( val * xi + b - MU( i ) ) ** 2
     a_11 = a_11 + val * val
     a_12 = a_12 + val
     a_22 = a_22 + 1.0_wp
     r_1 = r_1 + val * MU( i )
     r_2 = r_2 + MU( i )
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) f
   WRITE( 6, "( ' current mu, b = ', 2ES22.14 )" ) xi, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( ALPHA( i ) * xi + b - MU( i ) ) ** 2
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved mu, b = ', 2ES22.14 )" ) xi, b

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total

   pass = pass + 1
   IF ( pass <= pass_max .AND. f < f_best .AND. f > f_stop ) GO TO 10

!  end of loop

!  report final residuals

   f = 0.0_wp
   DO i = 1, obs
     val = 0.0_wp
     DO iy = 1, ny
       prod1 = Y( iy )
       DO ix = 1, nx
         prod2 = prod1 * X( ix )
         DO iv = 1, nv
           prod3 = prod2 * V( iv )
           DO iu = 1, nu
             prod4 = prod3 * U( iu )
             val = val + A( iu, iv, ix, iy, i ) * prod4
           END DO
         END DO
       END DO
     END DO
     f = f + ( val * xi + b - MU( i ) ) ** 2
   END DO

   WRITE( 6, "( /, 1X, 29( '=' ), ' pass limit ', 1X, 29( '=' ) )" )
   WRITE( 6, "( /, ' Optimal value =', ES22.14 )" ) 0.5_wp * f
!  WRITE( 6, "( ' Optimal U = ', /, ( 5ES12.4 ) )" ) U
!  WRITE( 6, "( ' Optimal V = ', /, ( 5ES12.4 ) )" ) V
!  WRITE( 6, "( ' Optimal X = ', /, ( 5ES12.4 ) )" ) X
!  WRITE( 6, "( ' Optimal Y = ', /, ( 5ES12.4 ) )" ) Y
   WRITE( 6, "( ' Optimal xi, b = ', /, 2ES22.14 )" ) xi, b
   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( ' total CPU time = ', F0.2, ' passes = ', I0 )" )              &
     time_total, pass

! tidy up afterwards

   CALL SLLS_terminate( data_u, control_u, inform_u )  !  delete workspace
   DEALLOCATE( p_u%B, p_u%X, p_u%Z )
   DEALLOCATE( p_u%Ao%val, p_u%Ao%type )

   CALL SLLS_terminate( data_v, control_v, inform_v )  !  delete workspace
   DEALLOCATE( p_v%B, p_v%X, p_v%Z )
   DEALLOCATE( p_v%Ao%val, p_v%Ao%type )

   CALL SLLS_terminate( data_x, control_x, inform_x )  !  delete workspace
   DEALLOCATE( p_x%B, p_x%X, p_x%Z )
   DEALLOCATE( p_x%Ao%val, p_x%Ao%type )

   CALL SLLS_terminate( data_y, control_y, inform_y )  !  delete workspace
   DEALLOCATE( p_y%B, p_y%X, p_y%Z )
   DEALLOCATE( p_y%Ao%val, p_y%Ao%type )

   END PROGRAM GALAHAD_SLLS_EXAMPLE6
