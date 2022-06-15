! THIS VERSION: GALAHAD 4.1 - 2022-06-12 AT 10:15 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE6

! use slls to solve a multilinear fitting problem
!  min 1/2 sum_i ( xi sum_iu sum_iv sum_ix sum_iy a(i,iu,iv,ix,iy)
!                  u_iu v_iv x_ix y_iy + b - mu_i)^2,
! where u, v, w and x lie in n and m regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with u free and v,x,y fixed, and vice versa

   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_RAND_double
   USE GALAHAD_NORMS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p_u, p_v, p_x, p_y
   TYPE ( SLLS_data_type ) :: data_u, data_v, data_x, data_y
   TYPE ( SLLS_control_type ) :: control_u, control_v, control_x, control_y
   TYPE ( SLLS_inform_type ) :: inform_u, inform_v, inform_x, inform_y
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: i, iu, iv, ix, iy, j, ne, pass
   TYPE ( RAND_seed ) :: seed
   REAL ( KIND = wp ) :: val, f, xi, b, a_11, a_12, a_22, r_1, r_2
   REAL ( KIND = wp ) :: alpha, beta, prod, prod1, prod2, prod3, prod4
   REAL :: time, time_start, time_total
   INTEGER, PARAMETER :: nu = 4
   INTEGER, PARAMETER :: nv = 8
   INTEGER, PARAMETER :: nx = 12
   INTEGER, PARAMETER :: ny = 16
   INTEGER, PARAMETER :: obs = 80
   INTEGER, PARAMETER :: pass_max = 3
!  LOGICAL, PARAMETER :: pertub_observations = .TRUE.
   LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   INTEGER :: U_stat( nu ), V_stat( nv ), X_stat( nx ), Y_stat( ny )
   REAL ( KIND = wp ) :: U( nu ), V( nv ), X( nx ), Y( nx ), MU( obs )
   REAL ( KIND = wp ) :: A( nu, nv, nx, ny, obs )

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
   CALL CPU_TIME( time_start )
   DO i = 1, obs
     DO iy = 1, ny
       DO ix = 1, nx
         DO iv = 1, nv
           DO iu = 1, nu
             CALL RAND_random_real( seed, .FALSE., A( iu, iv, ix, iy, i ) )
           END DO
         END DO
       END DO
     END DO
   END DO
   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( ' total CPU time = ', F0.2 )" ) time_total
   stop

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

   CALL SMT_put( p_u%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_u%A%val(obs * nu ), p_u%B( obs ), p_u%X( nu ) )
   p_u%m = obs ; p_u%n = nu ; p_u%A%m = p_u%m ; p_u%A%n = p_u%n

!  set up storage for the v problem

   CALL SMT_put( p_v%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_v%A%val(obs * nv ), p_v%B( obs ), p_v%X( nv ) )
   p_v%m = obs ; p_v%n = nv ; p_v%A%m = p_v%m ; p_v%A%n = p_v%n

!  set up storage for the x problem

   CALL SMT_put( p_x%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_x%A%val(obs * nx ), p_x%B( obs ), p_x%X( nx ) )
   p_x%m = obs ; p_x%n = nx ; p_x%A%m = p_x%m ; p_x%A%n = p_x%n

!  set up storage for the y problem

   CALL SMT_put( p_y%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_y%A%val(obs * ny ), p_y%B( obs ), p_y%X( ny ) )
   p_y%m = obs ; p_y%n = ny ; p_y%A%m = p_y%m ; p_y%A%n = p_y%n

!  starting point
   U = 1.0_wp / REAL( nu, KIND = wp ) ; V = 1.0_wp / REAL( nv, KIND = wp )
   X = 1.0_wp / REAL( nx, KIND = wp ) ; Y = 1.0_wp / REAL( ny, KIND = wp )
   xi = 2.0_wp ; b = 1.0_wp

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_u, control_u, inform_u )
   control_u%infinity = infinity
   control_u%print_level = 1
   control_u%exact_arc_search = .FALSE.
!  control_u%convert_control%print_level = 3

   CALL SLLS_initialize( data_v, control_v, inform_v )
   control_v%infinity = infinity
   control_v%print_level = 1
   control_v%exact_arc_search = .FALSE.
!  control_v%convert_control%print_level = 3

   CALL SLLS_initialize( data_x, control_x, inform_x )
   control_x%infinity = infinity
   control_x%print_level = 1
   control_x%exact_arc_search = .FALSE.
!  control_x%convert_control%print_level = 3

   CALL SLLS_initialize( data_y, control_y, inform_y )
   control_y%infinity = infinity
   control_y%print_level = 1
   control_y%exact_arc_search = .FALSE.
!  control_y%convert_control%print_level = 3

!  pass

   pass = 1
10 continue

   WRITE( 6, "( /, 1X, 32( '=' ), ' pass ', I0, 1X, 32( '=' ) )" ) pass

!  fix v, x, y, xi and b, and solve for u. Read in the rows of A

   ne = 0
   DO i = 1, obs
     prod = xi
     DO iy = 1, ny
       prod1 = prod * Y( iy )
       DO ix = 1, nx
         prod2 = prod1 * X( ix )
         DO iv = 1, nv
           prod3 = prod2 * V( iv )
           DO iu = 1, nu
             p_u%A%val( ne + iu ) = A( iu, iv, ix, iy, i ) * prod3
           END DO
         END DO
       END DO
     END DO
     ne = ne + nu
     p_u%B( i ) = MU( i ) - b
   END DO










   ne = 0
   DO i = 1, obs
     p_x%B( i ) = MU( i ) - b
   END DO
   p_x%X = X

!  solve the problem

   inform_x%status = 1
   CALL SLLS_solve( p_x, X_stat, data_x, control_x, inform_x, userdata )
   IF ( inform_x%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', /, ( 5ES12.4 ) )" )             &
     inform_x%iter, inform_x%obj, p_x%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform_x%status
     WRITE( 6, * ) inform_x%alloc_status, inform_x%bad_alloc
   END IF

!  record the new x

   X = p_x%X

!  fix x, xi and b, and solve for y. Read in the rows of A

   ne = 0
   DO i = 1, obs
     p_y%B( i ) = MU( i ) - b
   END DO
   p_y%X = Y

!  solve the problem

   inform_y%status = 1
   CALL SLLS_solve( p_y, Y_stat, data_y, control_y, inform_y, userdata )
   IF ( inform_y%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', /, ( 5ES12.4 ) )" )             &
     inform_y%iter, inform_y%obj, p_y%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform_y%status
     WRITE( 6, * ) inform_y%alloc_status, inform_y%bad_alloc
   END IF

!  record the new y

   Y = p_y%X

!  fix x and y, and solve for xi and b. Read in the rows of A

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     alpha = 1.0
     beta = 1.0_wp
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   val = a_11 * a_22 - a_12 * a_12

   WRITE( 6, "( ' new mu, b = ', 2ES12.4 )" ) &
     ( a_22 * r_1 - a_12 * r_2 ) / val, &
     ( - a_12 * r_1 + a_11 * r_2 ) / val

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  report final residuals

   val = 0.0_wp
   DO i = 1, obs
     val = val + ( xi + b - MU( i ) ) ** 2
   END DO
   WRITE( 6, "( /, ' 1/2|| F(x,y) - b||^2 = ', ES12.4 )" ) 0.5_wp * val

   pass = pass + 1
   IF ( pass <= pass_max .AND. SQRT( val ) > 0.00000001_wp ) GO TO 10

!  end of loop


! tidy up afterwards

   CALL SLLS_terminate( data_u, control_u, inform_u )  !  delete workspace
   DEALLOCATE( p_u%B, p_u%X, p_u%Z )
   DEALLOCATE( p_u%A%val, p_u%A%type )

   CALL SLLS_terminate( data_v, control_v, inform_v )  !  delete workspace
   DEALLOCATE( p_v%B, p_v%X, p_v%Z )
   DEALLOCATE( p_v%A%val, p_v%A%type )

   CALL SLLS_terminate( data_x, control_x, inform_x )  !  delete workspace
   DEALLOCATE( p_x%B, p_x%X, p_x%Z )
   DEALLOCATE( p_x%A%val, p_x%A%type )

   CALL SLLS_terminate( data_y, control_y, inform_y )  !  delete workspace
   DEALLOCATE( p_y%B, p_y%X, p_y%Z )
   DEALLOCATE( p_y%A%val, p_y%A%type )

   END PROGRAM GALAHAD_SLLS_EXAMPLE6
