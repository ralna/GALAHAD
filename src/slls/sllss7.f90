! THIS VERSION: GALAHAD 4.3 - 2023-12-31 AT 10:45 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE7

! use slls to solve a multilinear fitting problem
!  min 1/2 sum_i ( xi sum_il sum_ir sum_it sum_ip a(il,ir,it,ip,i)
!                  l_il r_ir theta_it y_phi + b - mu_i)^2,
! where u, v, w and x lie in n and o regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with u free and v,x,y fixed, and vice versa

   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_SLS_double
   USE GALAHAD_RAND_double
   USE GALAHAD_SPACE_double
   USE GALAHAD_NORMS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
   TYPE ( QPT_problem_type ) :: p_l, p_r, p_t, p_p
   TYPE ( SLLS_data_type ) :: data_l, data_r, data_t, data_p
   TYPE ( SLLS_control_type ) :: control_l, control_r, control_t, control_p
   TYPE ( SLLS_inform_type ) :: inform_l, inform_r, inform_t, inform_p
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: i, il, ir, it, ip, ix, iy, j, k, ll, ne, status, alloc_status
   INTEGER :: n, nl_free, nr_free, nt_free, np_free, iter, pass
!  INTEGER :: repeat
   REAL ( KIND = wp ) :: val, f, xi, b, dxi, db, a_11, a_12, a_22, r_1, r_2
   REAL ( KIND = wp ) :: alpha, beta, f_best, prod1, prod2, prod3, prod4
   REAL ( KIND = wp ) :: step, f_step, gts
   REAL :: time, time_start, time_total
!   INTEGER, PARAMETER :: nl = 5
!   INTEGER, PARAMETER :: nr = 4
!   INTEGER, PARAMETER :: nt = 3
!   INTEGER, PARAMETER :: np = 2
!   INTEGER, PARAMETER :: qx = 5
!   INTEGER, PARAMETER :: qy = 5

    INTEGER, PARAMETER :: nl = 18
    INTEGER, PARAMETER :: nr = 17
    INTEGER, PARAMETER :: nt = 16
    INTEGER, PARAMETER :: np = 15
    INTEGER, PARAMETER :: qx = 60
    INTEGER, PARAMETER :: qy = 60

   INTEGER, PARAMETER :: obs = qx * qy
   INTEGER, PARAMETER :: pass_max = 1000
   INTEGER, PARAMETER :: itref_max = 2
   LOGICAL, PARAMETER :: printi = .TRUE.
   LOGICAL, PARAMETER :: prints = .FALSE.
   LOGICAL, PARAMETER :: random_problem = .FALSE.
!  LOGICAL, PARAMETER :: random_problem = .TRUE.
   LOGICAL, PARAMETER :: pertub_observations = .TRUE.
!  LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   LOGICAL, PARAMETER :: solution = .TRUE.
!  LOGICAL, PARAMETER :: solution = .FALSE.
   LOGICAL, PARAMETER :: include_xi_and_b = .FALSE.
   REAL ( KIND = wp ), PARAMETER :: f_stop = ten ** ( - 16 )
   REAL ( KIND = wp ), PARAMETER :: scale = ten ** 15
   REAL ( KIND = wp ), PARAMETER :: pert = 0.001_wp
   REAL ( KIND = wp ), PARAMETER :: g_stop = ten ** ( - 10 )
!  REAL ( KIND = wp ), PARAMETER :: small = ten ** ( - 6 )
   LOGICAL, PARAMETER :: fresh_read = .FALSE.
   INTEGER :: L_stat( nl ), R_stat( nr ), T_stat( nt ), P_stat( np )
   INTEGER :: L_sold( nl ), R_sold( nr ), T_sold( nt ), P_sold( np )
   REAL ( KIND = wp ) :: L( nl ), R( nr ), THETA( nt ), PHI( np )
   REAL ( KIND = wp ) :: DL( nl ), DR( nr ), DTHETA( nt ), DPHI( np )
   REAL ( KIND = wp ) :: A_xi( obs ), A_b( obs )
   REAL ( KIND = wp ) :: MU( obs ), SIGMA( obs ), RHS( qx, qy )
   REAL ( KIND = wp ) :: SOL( obs + nl + nr + nt + np + 6 )
   REAL ( KIND = wp ) :: G( nl + nr + nt + np + 2 ), RES( obs )
   REAL ( KIND = wp ) :: A( nl, nr, nt, np, obs ), A_SAVE( obs )
   TYPE ( SMT_type ) :: MAT
   CHARACTER ( LEN = 80 ) :: array_name
   TYPE ( SLS_control_type ) :: SLS_control
   TYPE ( SLS_inform_type ) :: SLS_inform
   TYPE ( SLS_data_type ) :: SLS_data
   TYPE ( RAND_seed ) :: seed

!  start problem data

   CALL CPU_TIME( time_start )

!  generate a random problem

   IF ( random_problem ) THEN

!  compute a random solution

     CALL RAND_initialize( seed )
     CALL RAND_random_real( seed, .FALSE., L( : nl ) )
     L = MAX( L, 0.0_wp )
     val = SUM( L( : nl ) ) ; L( : nl ) = L( : nl ) / val
     CALL RAND_random_real( seed, .FALSE., R( : nr ) )
     R = MAX( R, 0.0_wp )
     val = SUM( R( : nr ) ) ; R( : nr ) = R( : nr ) / val
     CALL RAND_random_real( seed, .FALSE., THETA( : nt ) )
     THETA = MAX( THETA, 0.0_wp )
     val = SUM( THETA( : nt ) ) ; THETA( : nt ) = THETA( : nt ) / val
     CALL RAND_random_real( seed, .FALSE., PHI( : np ) )
     PHI = MAX( PHI, 0.0_wp )
     val = SUM( PHI( : np ) ) ; PHI( : np ) = PHI( : np ) / val
     xi = 2.0_wp ; b = 0.5_wp

     WRITE( 6, "( ' Optimal l = ', /, ( 5ES12.4 ) )" ) L
     WRITE( 6, "( ' Optimal r = ', /, ( 5ES12.4 ) )" ) R
     WRITE( 6, "( ' Optimal theta = ', /, ( 5ES12.4 ) )" ) THETA
     WRITE( 6, "( ' Optimal phi = ', /, ( 5ES12.4 ) )" ) PHI
     WRITE( 6, "( ' Optimal xi, b = ', /, 2ES22.14 )" ) xi / scale, b

!  random A

     DO i = 1, obs
       DO ip = 1, np
         DO it = 1, nt
           DO ir = 1, nr
             DO il = 1, nl
               CALL RAND_random_real( seed, .TRUE., A( il, ir, it, ip, i ) )
             END DO
           END DO
         END DO
       END DO
     END DO

!  generate a set of observations, mu_i with standard deviations of one

     DO i = 1, obs
       IF ( pertub_observations ) THEN
         CALL RAND_random_real( seed, .TRUE., val ) ; val = 0.0001 * val
       ELSE
         val = 0.0_wp
       END IF
       DO ip = 1, np
         prod1 = xi * PHI( ip )
         DO it = 1, nt
           prod2 = prod1 * THETA( it )
           DO ir = 1, nr
             prod3 = prod2 * R( ir )
             DO il = 1, nl
               val = val + A( il, ir, it, ip, i ) * prod3 * L( il )
             END DO
           END DO
         END DO
       END DO
       MU( i ) = val + b ; SIGMA( i ) = 1
     END DO

!  start from a perturbation of the solution

     IF ( solution ) THEN
       L = L + pert ; L = L / SUM( L )
       R = R + pert ; R = R / SUM( R )
       THETA = THETA + pert ; THETA = THETA / SUM( THETA )
       PHI = PHI + pert ; PHI = PHI / SUM( PHI )

!  start from the centre of the simplices

     ELSE
       L = 1.0_wp / REAL( nl, KIND = wp ) ; R = 1.0_wp / REAL( nr, KIND = wp )
       THETA = 1.0_wp / REAL( nt, KIND = wp )
       PHI = 1.0_wp / REAL( np, KIND = wp )
       xi = 3.0_wp ; b = 1.0_wp
     END IF

!  use data from a real problem

   ELSE

!  starting point

     IF ( solution ) THEN
       OPEN( unit = 23, file = "l.txt" )
       DO j = 1, nl
         READ( 23, * ) L( j )
       END DO
       CLOSE( unit = 24 )
       OPEN( unit = 24, file = "r.txt" )
       DO j = 1, nr
         READ( 24, * ) R( j )
       END DO
       CLOSE( unit = 24 )
       OPEN( unit = 25, file = "theta.txt" )
       DO j = 1, nt
         READ( 25, * ) THETA( j )
       END DO
       CLOSE( unit = 25 )
       OPEN( unit = 26, file = "phi.txt" )
       DO j = 1, np
         READ( 26, * ) PHI( j )
       END DO
       CLOSE( unit = 26 )
       xi = 2.351270579774912875d+3 ; b = 2.2d-4

       WRITE( 6, "( ' Optimal l = ', /, ( 5ES12.4 ) )" ) L
       WRITE( 6, "( ' Optimal r = ', /, ( 5ES12.4 ) )" ) R
       WRITE( 6, "( ' Optimal theta = ', /, ( 5ES12.4 ) )" ) THETA
       WRITE( 6, "( ' Optimal phi = ', /, ( 5ES12.4 ) )" ) PHI
       WRITE( 6, "( ' Optimal xi, b = ', /, 2ES22.14 )" ) xi / scale, b

       L = L + pert ; L = L / SUM( L )
       R = R + pert ; R = R / SUM( R )
       THETA = THETA + pert ; THETA = THETA / SUM( THETA )
       PHI = PHI + pert ; PHI = PHI / SUM( PHI )

     ELSE
       L = 1.0_wp / REAL( nl, KIND = wp ) ; R = 1.0_wp / REAL( nr, KIND = wp )
       THETA = 1.0_wp / REAL( nt, KIND = wp )
       PHI = 1.0_wp / REAL( np, KIND = wp )
       xi = 3.0_wp ; b = 1.0_wp
     END IF
!    R = R + 0.001_wp
!    L = L + 0.001_wp
!    THETA = THETA + 0.001_wp
!    PHI = PHI + 0.001_wp

!  input observations mu (and set sigma to be mu)

     OPEN( unit = 21, file = "intensities.txt" )
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

     IF ( fresh_read ) THEN
       OPEN( unit = 22, file = "G.txt" )
       j = 0
       DO
         READ( 22, *, END = 1 ) ix, iy, il, ir, it, ip, val
         i = qx * iy + ix + 1
         A( il + 1, ir + 1, it + 1, ip + 1, i ) = val / ( scale * SIGMA( i ) )
         j = j + 1
         IF ( MOD( j, 10000000 ) == 0 ) THEN
           CALL CPU_TIME( time ) ; time_total = time - time_start
           WRITE( 6, "( ' line ', I0, ' read, CPU time so far = ', F0.2 )" )   &
            j, time_total
         END IF
       END DO
     1 CONTINUE
       CLOSE( unit = 22 )

       CALL CPU_TIME( time ) ; time_total = time - time_start
       WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total
       WRITE( 6, "( 1X, I0, ' lines read' )" ) j
!      time_start = time
!      WRITE( 21, * ) A
!      CALL CPU_TIME( time ) ; time_total = time - time_start
!      WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total
!      time_start = time
!      REWIND 21
!      READ( 21, * ) A
!      CALL CPU_TIME( time ) ; time_total = time - time_start
!      WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total
!     stop
     ELSE
       OPEN( unit = 22, file = "A.txt" )
       READ( 22, * ) A
       CLOSE( unit = 22 )
       CALL CPU_TIME( time ) ; time_total = time - time_start
       WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total
     END IF
   END IF
   write(6,*) ' max A = ', MAXVAL( ABS( A ) )

!  set up storage for the l problem

   CALL SMT_put( p_l%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_l%Ao%val( obs * nl ), p_l%B( obs ), p_l%X( nl ) )
   p_l%o = obs ; p_l%n = nl ; p_l%Ao%m = p_l%o ; p_l%Ao%n = p_l%n

!  set up storage for the r problem

   CALL SMT_put( p_r%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_r%Ao%val( obs * nr ), p_r%B( obs ), p_r%X( nr ) )
   p_r%o = obs ; p_r%n = nr ; p_r%Ao%m = p_r%o ; p_r%Ao%n = p_r%n

!  set up storage for the theta problem

   CALL SMT_put( p_t%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_t%Ao%val( obs * nt ), p_t%B( obs ), p_t%X( nt ) )
   p_t%o = obs ; p_t%n = nt ; p_t%Ao%m = p_t%o ; p_t%Ao%n = p_t%n

!  set up storage for the phi problem

   CALL SMT_put( p_p%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_p%Ao%val( obs * np ), p_p%B( obs ), p_p%X( np ) )
   p_p%o = obs ; p_p%n = np ; p_p%Ao%m = p_p%o ; p_p%Ao%n = p_p%n

!  set up storage type for B

   array_name = 'sllss7: MAT%type'
   CALL SMT_put( MAT%type, 'SPARSE_BY_ROWS', alloc_status )
   IF ( alloc_status /= 0 ) GO TO 99
!  CALL SLS_initialize( 'ma97', sls_data, sls_control, sls_inform )
   CALL SLS_initialize( 'ma57', sls_data, sls_control, sls_inform )
!  CALL SLS_initialize( 'sils', sls_data, sls_control, sls_inform )

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_l, control_l, inform_l )
   IF ( printi ) THEN
     control_l%print_level = 1
   ELSE
     control_l%print_level = 0
   END IF
   control_l%exact_arc_search = .FALSE.
!  control_l%convert_control%print_level = 3
   control_l%stop_d = ten ** ( - 10 )

   CALL SLLS_initialize( data_r, control_r, inform_r )
   IF ( printi ) THEN
     control_r%print_level = 1
   ELSE
     control_r%print_level = 0
   END IF
   control_r%exact_arc_search = .FALSE.
!  control_r%convert_control%print_level = 3
   control_r%stop_d = ten ** ( - 10 )

   CALL SLLS_initialize( data_t, control_t, inform_t )
   IF ( printi ) THEN
     control_t%print_level = 1
   ELSE
     control_t%print_level = 0
   END IF
   control_t%exact_arc_search = .FALSE.
!  control_t%convert_control%print_level = 3
   control_t%stop_d = ten ** ( - 10 )

   CALL SLLS_initialize( data_p, control_p, inform_p )
   IF ( printi ) THEN
     control_p%print_level = 1
   ELSE
     control_p%print_level = 0
   END IF
   control_p%exact_arc_search = .FALSE.
!  control_p%convert_control%print_level = 3
   control_p%stop_d = ten ** ( - 10 )

   CALL CPU_TIME( time_start )

!  fix l, r, theta and phi, and solve for xi and b

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
             alpha = alpha + A( il, ir, it, ip, i ) * prod3 * L( il )
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
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
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
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total

   WHERE ( L > 0.0_wp ) ; L_stat = 0 ; ELSEWHERE ; L_stat = - 1 ; END WHERE
   WHERE ( R > 0.0_wp ) ; R_stat = 0 ; ELSEWHERE ; R_stat = - 1 ; END WHERE
   WHERE ( THETA > 0.0_wp ) ; T_stat = 0 ; ELSEWHERE ; T_stat = - 1 ; END WHERE
   WHERE ( PHI > 0.0_wp ) ; P_stat = 0 ; ELSEWHERE ; P_stat = - 1 ; END WHERE

!  pass

   pass = 1
10 CONTINUE
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
         prod1 = PHI( ip )
         DO it = 1, nt
           prod2 = prod1 * THETA( it )
           DO ir = 1, nr
             val = val + A( il, ir, it, ip, i ) * prod2 * R( ir )
           END DO
         END DO
       END DO
       p_l%Ao%val( ne + il ) = val
     END DO
     ne = ne + nl
   END DO

!  before finding l, fix l, r, theta and phi, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   ne = 0
   DO i = 1, obs
     alpha = DOT_PRODUCT( L, p_l%Ao%val( ne + 1 : ne + nl ) )
     ne = ne + nl
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
     p_l%B( i ) = MU( i ) - b / SIGMA( i )
   END DO
   f = 0.5_wp * f
   p_l%Ao%val = xi * p_l%Ao%val
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

!  solve the l problem

   inform_l%status = 1
   L_sold = L_stat
   CALL SLLS_solve( p_l, L_stat, data_l, control_l, inform_l, userdata )
   IF ( inform_l%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new l', /,                &
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
         prod1 = PHI( ip )
         DO it = 1, nt
           prod2 = prod1 * THETA( it )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_r%Ao%val( ne + ir ) = val
     END DO
     ne = ne + nr
   END DO

!  before finding r, fix l, r, theta and phi, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   ne = 0
   DO i = 1, obs
     alpha = DOT_PRODUCT( R, p_r%Ao%val( ne + 1 : ne + nr ) )
     ne = ne + nr
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
     p_r%B( i ) = MU( i ) - b / SIGMA( i )
   END DO
   f = 0.5_wp * f
   p_r%Ao%val = xi * p_r%Ao%val
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

!  solve the v problem

   inform_r%status = 1
   R_sold = R_stat
   CALL SLLS_solve( p_r, R_stat, data_r, control_r, inform_r, userdata )
   IF ( inform_r%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new r', /,                &
    &     ' Optimal objective value =', ES22.14 )" ) inform_r%iter, inform_r%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_r%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_r%status
     WRITE( 6, * ) inform_r%alloc_status, inform_r%bad_alloc
   END IF

!  record the new r

   R = p_r%X

!  fix l, r, phi, xi and b, and solve for theta

!DO repeat = 1, 2
   p_t%X = THETA

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO it = 1, nt
       val = 0.0_wp
       DO ip = 1, np
         prod1 = PHI( ip )
         DO ir = 1, nr
           prod2 = prod1 * R( ir )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_t%Ao%val( ne + it ) = val
     END DO
     ne = ne + nt
   END DO

!  before finding theta, fix l, r, theta and phi, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   ne = 0
   DO i = 1, obs
     alpha = DOT_PRODUCT( THETA, p_t%Ao%val( ne + 1 : ne + nt ) )
     ne = ne + nt
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
     p_t%B( i ) = MU( i ) - b / SIGMA( i )
   END DO
   f = 0.5_wp * f
   p_t%Ao%val = xi * p_t%Ao%val
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

!  solve the theta problem

   inform_t%status = 1
   T_sold = T_stat
   CALL SLLS_solve( p_t, T_stat, data_t, control_t, inform_t, userdata )
   IF ( inform_t%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new theta', /,            &
    &     ' Optimal objective value =', ES22.14 )" ) inform_t%iter, inform_t%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_t%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_t%status
     WRITE( 6, * ) inform_t%alloc_status, inform_t%bad_alloc
   END IF

!  record the new theta

   THETA = p_t%X
!END DO

!  fix l, r, theta, xi and b, and solve for phi

   p_p%X = PHI

!  set the rows of A and observations

   ne = 0
   DO i = 1, obs
     DO ip = 1, np
       val = 0.0_wp
       DO it = 1, nt
         prod1 = THETA( it )
         DO ir = 1, nr
           prod2 = prod1 * R( ir )
           DO il = 1, nl
             val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
           END DO
         END DO
       END DO
       p_p%Ao%val( ne + ip ) = val
     END DO
     ne = ne + np
   END DO

!  before finding phi, fix l, r, theta and phi, and solve for xi and b

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   ne = 0
   DO i = 1, obs
     alpha = DOT_PRODUCT( PHI, p_p%Ao%val( ne + 1 : ne + np ) )
     ne = ne + np
     A_SAVE( i ) = alpha ; beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
   WRITE( 6, "( /, ' current obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
     p_p%B( i ) = MU( i ) - b / SIGMA( i )
   END DO
   f = 0.5_wp * f
   p_p%Ao%val = xi * p_p%Ao%val
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

!  solve the phi problem

   inform_p%status = 1
   P_sold = P_stat
   CALL SLLS_solve( p_p, P_stat, data_p, control_p, inform_p, userdata )
   IF ( inform_p%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations for new phi', /,              &
    &     ' Optimal objective value =', ES22.14 )" ) inform_p%iter, inform_p%obj
     IF ( prints )WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_p%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve u exit status = ', I0 ) " ) inform_p%status
     WRITE( 6, * ) inform_p%alloc_status, inform_p%bad_alloc
   END IF

!  record the new phi

   PHI = p_p%X

!  fix l, r, theta and phi, and solve for xi and b

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
             alpha = alpha + A( il, ir, it, ip, i ) * prod3 * L( il )
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
   WRITE( 6, "( ' current xi, b = ', 2ES22.14 )" ) xi / scale, b
   val = a_11 * a_22 - a_12 * a_12

!  record new xi and b

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  compute improved objective

   f = 0.0_wp
   DO i = 1, obs
     f = f + ( A_SAVE( i ) * xi + b / SIGMA( i ) - MU( i ) ) ** 2
     p_t%B( i ) = MU( i ) - b / SIGMA( i )
   END DO
   f = 0.5_wp * f
   WRITE( 6, "( ' improved obj =', ES22.14 )" ) f
   WRITE( 6, "( ' improved xi, b = ', 2ES22.14 )" ) xi / scale, b

!  solve the system
!
!  (  I   A  0 ) ( r )   ( b )
!  ( A^T  0  E ) ( s ) = ( 0 )
!  (  0  E^T 0 ) ( u )   ( 0 )
!
!  where A = (  A_l  :  A_r  :  A_theta  :  A_phi  : A_xi : A_b ),
!
!            ( e_l^T :   0   :     0     :    0    :   0  :  0  )
!      E^T = (   0   : e_r^T :     0     :    0    :   0  :  0  )
!            (   0   :   0   : e_theta^T :    0    :   0  :  0  )
!            (   0   :   0   :     0     : e_phi^T :   0  :  0  )
!
!  and A_l (etc) are the columns corresponding to free variables
!
!   only the lower triangle of the coefficient matrix MAT is stored

!  compute the number of free variables for each component

   nl_free = COUNT( L_stat == 0 )
   nr_free = COUNT( R_stat == 0 )
   nt_free = COUNT( T_stat == 0 )
   np_free = COUNT( P_stat == 0 )

   WRITE( 6, "( ' ==== free/nl = ', I0, '/', I0, &
  &                  ' free/nr = ', I0, '/', I0, &
  &                  ' free/nt = ', I0, '/', I0, &
  &                  ' free/np = ', I0, '/', I0 )" ) &
     nl_free, nl, nr_free, nr, nt_free, nt, np_free, np

   IF ( COUNT( L_stat /= L_sold ) + COUNT( R_stat /= R_sold ) +                &
        COUNT( T_stat /= T_sold ) + COUNT( P_stat /= P_sold ) == 0 ) THEN

!   WHERE ( L > small ) ; L_stat = 0
!   ELSEWHERE ; L = 0.0_wp ; L_stat = - 1 ; END WHERE
!   WHERE ( R > small ) ; R_stat = 0
!   ELSEWHERE ; R = 0.0_wp ; R_stat = - 1 ; END WHERE
!   WHERE ( THETA > small ) ; T_stat = 0
!   ELSEWHERE ; THETA = 0.0_wp ; T_stat = - 1 ; END WHERE
!   WHERE ( PHI > small ) ; P_stat = 0
!   ELSEWHERE ; PHI = 0.0_wp ; P_stat = - 1 ; END WHERE

   CALL CPU_TIME( time ) ; time_total = time - time_start
!  WRITE( 6, "( /, ' CPU time start = ', F0.2 )" ) time_total

!  compute the array dimensions for MAT

     IF ( include_xi_and_b ) THEN
       n = obs + nl_free + nr_free + nt_free + np_free + 6 ; MAT%n = n
       MAT%ne = obs * ( nl_free + nr_free + nt_free + np_free + 3 ) +          &
                nl_free + nr_free + nt_free + np_free
     ELSE
       n = obs + nl_free + nr_free + nt_free + np_free + 4 ; MAT%n = n
       MAT%ne = obs * ( nl_free + nr_free + nt_free + np_free + 1 ) +          &
                nl_free + nr_free + nt_free + np_free
     END IF
     WRITE( 6, "( ' ==== n, ne = ', I0, 1X, I0 )" ) MAT%n, MAT%ne

!  allocate space for MAT

     array_name = 'sllss7: MAT%val'
     CALL SPACE_resize_array( MAT%ne, MAT%val, status, alloc_status,           &
            array_name = array_name, exact_size = .FALSE. )
     IF ( status /= 0 ) GO TO 99

     array_name = 'sllss7: MAT%col'
     CALL SPACE_resize_array( MAT%ne, MAT%col, status, alloc_status,           &
            array_name = array_name, exact_size = .FALSE. )
     IF ( status /= 0 ) GO TO 99

     array_name = 'sllss7: MAT%ptr'
     CALL SPACE_resize_array( MAT%n + 1, MAT%ptr, status, alloc_status,        &
            array_name = array_name, exact_size = .FALSE. )
     IF ( status /= 0 ) GO TO 99

!  loop to improve the objective by Gauss-Newton steps

!    DO iter = 1, 5
     DO iter = 1, 1
!    DO iter = 1, 0

!  construct the constituent parts of A

!  parts from A_l

       ne = 0
       DO i = 1, obs
         DO il = 1, nl
           IF ( L_stat( il ) /= 0 ) CYCLE
           val = 0.0_wp
           DO ip = 1, np
             IF ( P_stat( ip ) /= 0 ) CYCLE
             prod1 = PHI( ip )
             DO it = 1, nt
               IF ( T_stat( it ) /= 0 ) CYCLE
               prod2 = prod1 * THETA( it )
               DO ir = 1, nr
                 IF ( R_stat( ir ) /= 0 ) CYCLE
                 val = val + A( il, ir, it, ip, i ) * prod2 * R( ir )
               END DO
             END DO
           END DO
           p_l%Ao%val( ne + il ) = val
         END DO
         ne = ne + nl
       END DO

   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( ' CPU time A_l = ', F0.2 )" ) time_total

!  parts from A_r

       ne = 0
       DO i = 1, obs
         DO ir = 1, nr
           IF ( R_stat( ir ) /= 0 ) CYCLE
           val = 0.0_wp
           DO ip = 1, np
             IF ( P_stat( ip ) /= 0 ) CYCLE
             prod1 = PHI( ip )
             DO it = 1, nt
               IF ( T_stat( it ) /= 0 ) CYCLE
               prod2 = prod1 * THETA( it )
               DO il = 1, nl
                 IF ( L_stat( il ) /= 0 ) CYCLE
                 val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
               END DO
             END DO
           END DO
           p_r%Ao%val( ne + ir ) = val
         END DO
         ne = ne + nr
       END DO

   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( ' CPU time A_r = ', F0.2 )" ) time_total

!  parts from A_theta

       ne = 0
       DO i = 1, obs
         DO it = 1, nt
           IF ( T_stat( it ) /= 0 ) CYCLE
           val = 0.0_wp
           DO ip = 1, np
             IF ( P_stat( ip ) /= 0 ) CYCLE
             prod1 = PHI( ip )
             DO ir = 1, nr
               IF ( R_stat( ir ) /= 0 ) CYCLE
               prod2 = prod1 * R( ir )
               DO il = 1, nl
                 IF ( L_stat( il ) /= 0 ) CYCLE
                 val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
               END DO
             END DO
           END DO
           p_t%Ao%val( ne + it ) = val
         END DO
         ne = ne + nt
       END DO

   CALL CPU_TIME( time ) ; time_total = time - time_start
 !  WRITE( 6, "( ' CPU time A_theta = ', F0.2 )" ) time_total

!  parts from A_phi

       ne = 0
       DO i = 1, obs
         DO ip = 1, np
           IF ( P_stat( ip ) /= 0 ) CYCLE
           val = 0.0_wp
           DO it = 1, nt
             IF ( T_stat( it ) /= 0 ) CYCLE
             prod1 = THETA( it )
             DO ir = 1, nr
               IF ( R_stat( ir ) /= 0 ) CYCLE
               prod2 = prod1 * R( ir )
               DO il = 1, nl
                 IF ( L_stat( il ) /= 0 ) CYCLE
                 val = val + A( il, ir, it, ip, i ) * prod2 * L( il )
               END DO
             END DO
           END DO
           p_p%Ao%val( ne + ip ) = val
         END DO
         ne = ne + np
       END DO

   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( /, ' CPU time A_phi = ', F0.2 )" ) time_total

!  parts from A_xi and A_b, as well as the residuals

       DO i = 1, obs
         alpha = 0.0_wp
         DO ip = 1, np
           IF ( P_stat( ip ) /= 0 ) CYCLE
           prod1 = PHI( ip )
           DO it = 1, nt
             IF ( T_stat( it ) /= 0 ) CYCLE
             prod2 = prod1 * THETA( it )
             DO ir = 1, nr
               IF ( R_stat( ir ) /= 0 ) CYCLE
               prod3 = prod2 * R( ir )
               DO il = 1, nl
                 IF ( L_stat( il ) /= 0 ) CYCLE
                 alpha = alpha + A( il, ir, it, ip, i ) * prod3 * L( il )
               END DO
             END DO
           END DO
         END DO
         IF ( include_xi_and_b ) THEN
           A_xi( i ) = alpha ; A_b( i ) = 1.0_wp / SIGMA( i )
         END IF
         RES( i ) = alpha * xi + b / SIGMA( i )- MU( i )
       END DO
       SOL( 1 : obs ) = - RES( 1 : obs )
       SOL( obs + 1 : n ) = 0.0_wp

   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( ' CPU time A = ', F0.2 )" ) time_total

!  compute G = A^T res

IF ( .FALSE. ) THEN
     G = 0.0_wp
     ne = 0 ; k = 0
     DO i = 1, obs
       G( k + 1 : k + nl ) =                                                   &
         G( k + 1 : k + nl ) + p_l%Ao%val( ne + 1 : ne + nl ) * RES( i )
       ne = ne + nl
     END DO
     k = k + nl

     ne = 0
     DO i = 1, obs
       G( k + 1 : k + nr ) =                                                   &
         G( k + 1 : k + nr ) + p_r%Ao%val( ne + 1 : ne + nr ) * RES( i )
       ne = ne + nr
     END DO
     k = k + nr

     ne = 0
     DO i = 1, obs
       G( k + 1 : k + nt ) =                                                   &
         G( k + 1 : k + nt ) + p_t%Ao%val( ne + 1 : ne + nt ) * RES( i )
       ne = ne + nt
     END DO
     k = k + nt

     ne = 0
     DO i = 1, obs
       G( k + 1 : k + np ) =                                                   &
         G( k + 1 : k + np ) + p_p%Ao%val( ne + 1 : ne + np ) * RES( i )
       ne = ne + np
     END DO
     k = k + np + 1

     DO i = 1, obs
       G( k ) = G( k ) + A_xi( i ) * RES( i )
     END DO
     k = k + 1

     DO i = 1, obs
       G( k ) = G( k ) + A_b( i ) * RES( i )
     END DO
END IF

!  fill MAT

!  the identity block

       DO j = 1, obs
         MAT%ptr( j ) = j ; MAT%col( j ) = j ; MAT%val( j ) = 1.0_wp
       END DO

!  the ( A^T 0 ) block

       ll = obs ; ne = obs + 1

!  l components

       DO j = 1, nl
         IF ( L_stat( j ) == 0 ) THEN
           ll = ll + 1 ; MAT%ptr( ll ) = ne ; k = j
           DO i = 1, obs
             MAT%col( ne ) = i ; MAT%val( ne ) =  p_l%Ao%val( k )
             k = k + nl
             ne = ne + 1
           END DO
         END IF
       END DO

!  r components

       DO j = 1, nr
         IF ( R_stat( j ) == 0 ) THEN
           ll = ll + 1 ; MAT%ptr( ll ) = ne ; k = j
           DO i = 1, obs
             MAT%col( ne ) = i ; MAT%val( ne ) =  p_r%Ao%val( k )
             k = k + nr
             ne = ne + 1
           END DO
         END IF
       END DO

!  theta components

       DO j = 1, nt
         IF ( T_stat( j ) == 0 ) THEN
           ll = ll + 1 ; MAT%ptr( ll ) = ne ; k = j
           DO i = 1, obs
             MAT%col( ne ) = i ; MAT%val( ne ) =  p_t%Ao%val( k )
             k = k + nt
             ne = ne + 1
           END DO
         END IF
       END DO

!  phi components

       DO j = 1, np
         IF ( P_stat( j ) == 0 ) THEN
           ll = ll + 1 ; MAT%ptr( ll ) = ne ; k = j
           DO i = 1, obs
             MAT%col( ne ) = i ; MAT%val( ne ) =  p_p%Ao%val( k )
             k = k + np
             ne = ne + 1
           END DO
         END IF
       END DO

!  xi components

       IF ( include_xi_and_b ) THEN
         ll = ll + 1 ; MAT%ptr( ll ) = ne
         DO i = 1, obs
           MAT%col( ne ) = i ; MAT%val( ne ) =  A_xi( i )
           ne = ne + 1
         END DO

!  b components

         ll = ll + 1 ; MAT%ptr( ll ) = ne
         DO i = 1, obs
           MAT%col( ne ) = i ; MAT%val( ne ) =  A_b( i )
           ne = ne + 1
         END DO
       END IF

!  the ( 0 E^T ) block

       i = obs

!  l components

       ll = ll + 1 ; MAT%ptr( ll ) = ne
       DO j = 1, nl
         IF ( L_stat( j ) == 0 ) THEN
           i = i + 1
           MAT%col( ne ) = i ; MAT%val( ne ) = 1.0_wp
           ne = ne + 1
         END IF
       END DO

!  r components

       ll = ll + 1 ; MAT%ptr( ll ) = ne
       DO j = 1, nr
         IF ( R_stat( j ) == 0 ) THEN
           i = i + 1
           MAT%col( ne ) = i ; MAT%val( ne ) = 1.0_wp
           ne = ne + 1
         END IF
       END DO

!  theta components

       ll = ll + 1 ; MAT%ptr( ll ) = ne
       DO j = 1, nt
         IF ( T_stat( j ) == 0 ) THEN
           i = i + 1
           MAT%col( ne ) = i ; MAT%val( ne ) = 1.0_wp
           ne = ne + 1
         END IF
       END DO

!  phi components

       ll = ll + 1 ; MAT%ptr( ll ) = ne
       DO j = 1, np
         IF ( P_stat( j ) == 0 ) THEN
           i = i + 1
           MAT%col( ne ) = i ; MAT%val( ne ) = 1.0_wp
           ne = ne + 1
         END IF
       END DO
       ll = ll + 1 ; MAT%ptr( ll ) = ne

   CALL CPU_TIME( time ) ; time_total = time - time_start
 !  WRITE( 6, "( ' CPU time mat = ', F0.2 )" ) time_total

!do i = 1, n
!  write( 6, "( ' row = ', I6, ' ( col, val) =', / ( 4( I8, ES12.4 ) ) )" ) &
!    i, ( MAT%col( k ), MAT%val( k ), k = MAT%ptr( i ), MAT%ptr( i + 1 ) - 1 )
!end do
!stop

!write(6,"( ' n, ll, ne, ptr(n+1)-1,i = ', 5I8 )" ) n, ll-1, MAT%ne, ne-1, i

!  analyse the structure of the augmented matrix

       SLS_control%max_iterative_refinements = itref_max
       SLS_control%print_level = 3
       CALL SLS_analyse( MAT, sls_data, sls_control, sls_inform )

!  test for analysis failure

       IF ( sls_inform%status < 0 ) THEN
         WRITE( 6, '( A, I0 )' )                                               &
            ' Failure of SLS_analyse with status = ', sls_inform%status
         STOP
       END IF

!  factorize the augmented matrix

     CALL SLS_factorize( MAT, sls_data, sls_control, sls_inform )

!  test for factorization failure

       IF ( sls_inform%status < 0 ) THEN
         WRITE( 6, '( A, I0 )' )                                               &
            ' Failure of SLS_factorize with status = ', sls_inform%status
         STOP
       END IF
   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( ' CPU time fact = ', F0.2 )" ) time_total

!  solve system

       WRITE( 6, "( ' ||rhs|| =', ES10.2 )" ) TWO_NORM( SOL( : n ) )
       CALL SLS_solve( MAT, SOL, sls_data, sls_control, sls_inform )
       WRITE( 6, "( ' ||sol|| =', ES10.2 )" ) TWO_NORM( SOL( : n ) )

!  extract the solution, and compute the maximum allowed step size

       ll = obs ; step = 1.0_wp ; k = 0

!  l components

!      gts = 0.0_wp
       DO j = 1, nl
         k = k + 1
         IF ( L_stat( j ) == 0 ) THEN
           ll = ll + 1
           val = SOL( ll )
           DL( j ) = val
!write(6,"( ' l, dl = ', 2ES12.4 )" ) L(j), DL(j)
           IF ( val < 0.0_wp ) step = MIN( step, - L( j ) / val )
!        gts = gts + val * G( k )
         ELSE
           DL( j ) = 0.0_wp
         END IF
       END DO

!  r components

       DO j = 1, nr
         k = k + 1
         IF ( R_stat( j ) == 0 ) THEN
           ll = ll + 1
           val = SOL( ll )
           DR( j ) = val
!write(6,"( ' r, dr = ', 2ES12.4 )" ) R(j), DR(j)
           IF ( val < 0.0_wp ) step = MIN( step, - R( j ) / val )
!        gts = gts + val * G( k )
         ELSE
           DR( j ) = 0.0_wp
         END IF
       END DO

!  theta components

       DO j = 1, nt
         k = k + 1
         IF ( T_stat( j ) == 0 ) THEN
           ll = ll + 1
           val = SOL( ll )
           DTHETA( j ) = val
!write(6,"( ' theta, dtheta = ', 2ES12.4 )" ) THETA(j), DTHETA(j)
           IF ( val < 0.0_wp ) step = MIN( step, - THETA( j ) / val )
!          gts = gts + val * G( k )
         ELSE
           DTHETA( j ) = 0.0_wp
         END IF
       END DO

!  phi components

       DO j = 1, np
         k = k + 1
         IF ( P_stat( j ) == 0 ) THEN
           ll = ll + 1
           val = SOL( ll )
           DPHI( j ) = val
  !write(6,"( ' phi, dphi = ', 2ES12.4 )" ) PHI(j), DPHI(j)
           IF ( val < 0.0_wp ) step = MIN( step, - PHI( j ) / val )
!         gts = gts + val * G( k )
         ELSE
           DPHI( j ) = 0.0_wp
         END IF
       END DO
   CALL CPU_TIME( time ) ; time_total = time - time_start
!   WRITE( 6, "( ' CPU time sol = ', F0.2 )" ) time_total

!  xi and b components

       IF ( include_xi_and_b ) THEN
         dxi = SOL( ll + 1 ) ; db = SOL( ll + 2 )
       END IF
!      gts = gts + dxi * G( k + 1 ) + db * G( k + 2 )

       WRITE( 6, "( ' dl = ', /, ( 5ES12.4 ) )" ) DL
       WRITE( 6, "( ' dr = ', /, ( 5ES12.4 ) )" ) DR
       WRITE( 6, "( ' dtheta = ', /, ( 5ES12.4 ) )" ) DTHETA
       WRITE( 6, "( ' dphi = ', /, ( 5ES12.4 ) )" ) DPHI
       IF ( include_xi_and_b )                                                 &
         WRITE( 6, "( ' dxi, db = ', /, 2ES22.14 )" ) dxi, db
       WRITE( 6, "( ' max steplength =', ES10.2 )" ) step
!      WRITE( 6, "( ' gts =', ES10.2 )" ) gts
       gts =  DOT_PRODUCT( RES( : obs ), RES( : obs ) ) +                      &
              DOT_PRODUCT( RES( : obs ), SOL( : obs ) )
       WRITE( 6, "( ' gts =', ES10.2 )" ) gts

!  compute the new objective value

       DO
         f_step = 0.0_wp
         DO i = 1, obs
           alpha = 0.0_wp
           DO ip = 1, np
             prod1 = PHI( ip ) + step * DPHI( ip )
             DO it = 1, nt
               prod2 = prod1 * ( THETA( it ) + step * DTHETA( it ) )
               DO ir = 1, nr
                 prod3 = prod2 * ( R( ir ) + step * DR( ir ) )
                 DO il = 1, nl
                   prod4 = prod3 * ( L( il ) + step * DL( il ) )
                   alpha = alpha + A( il, ir, it, ip, i ) * prod4
                 END DO
               END DO
             END DO
           END DO
           IF ( include_xi_and_b ) THEN
             f_step = f_step + ( alpha * ( xi + step * dxi )                   &
                             + ( b + step * db ) / SIGMA( i )- MU( i ) ) ** 2
           ELSE
             f_step = f_step + ( alpha * xi + b / SIGMA( i )- MU( i ) ) ** 2
           END IF
         END DO
         f_step = 0.5_wp * f_step
         WRITE( 6, "( ' new f =', ES22.14 )" ) f_step
         IF ( f_step <= f ) THEN
           L = L + step * DL
           R = R + step * DR
           THETA = THETA + step * DTHETA
           PHI = PHI + step * DPHI
           IF ( include_xi_and_b ) THEN
             xi = xi + step * dxi
             b = b + step * db
           END IF
           f = f_step
           EXIT
         ELSE
!        EXIT
         END IF
         step = 0.5_wp * step
       END DO

!  check for termination

       IF ( gts < g_stop ) GO TO 90
     END DO
   END IF

   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( /, ' CPU time so far = ', F0.2 )" ) time_total

   pass = pass + 1
   IF ( pass <= pass_max .AND. f < f_best .AND. f > f_stop ) GO TO 10

!  end of loop

 90 CONTINUE

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
             alpha = alpha + A( il, ir, it, ip, i ) * prod3 * L( il )
           END DO
         END DO
       END DO
     END DO
     f = f + ( alpha * xi + b / SIGMA( i )- MU( i ) ) ** 2
   END DO
   f = 0.5_wp * f

   WRITE( 6, "( /, 1X, 29( '=' ), ' pass limit ', 1X, 29( '=' ) )" )
   WRITE( 6, "( ' Optimal l = ', /, ( 5ES12.4 ) )" ) L
   WRITE( 6, "( ' Optimal r = ', /, ( 5ES12.4 ) )" ) R
   WRITE( 6, "( ' Optimal theta = ', /, ( 5ES12.4 ) )" ) THETA
   WRITE( 6, "( ' Optimal phi = ', /, ( 5ES12.4 ) )" ) PHI
   WRITE( 6, "( ' Optimal xi, b = ', /, 2ES22.14 )" ) xi / scale, b
   WRITE( 6, "( /, ' Optimal value =', ES22.14 )" ) f
   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( ' total CPU time = ', F0.2, ' passes = ', I0 )" )              &
     time_total, pass

! tidy up afterwards

   CALL SLLS_terminate( data_l, control_l, inform_l )  !  delete workspace
   DEALLOCATE( p_l%B, p_l%X, p_l%Z )
   DEALLOCATE( p_l%Ao%val, p_l%Ao%type )

   CALL SLLS_terminate( data_r, control_r, inform_r )  !  delete workspace
   DEALLOCATE( p_r%B, p_r%X, p_r%Z )
   DEALLOCATE( p_r%Ao%val, p_r%Ao%type )

   CALL SLLS_terminate( data_t, control_t, inform_t )  !  delete workspace
   DEALLOCATE( p_t%B, p_t%X, p_t%Z )
   DEALLOCATE( p_t%Ao%val, p_t%Ao%type )

   CALL SLLS_terminate( data_p, control_p, inform_p )  !  delete workspace
   DEALLOCATE( p_p%B, p_p%X, p_p%Z )
   DEALLOCATE( p_p%Ao%val, p_p%Ao%type )

   DEALLOCATE( MAT%val, MAT%col, MAT%ptr, MAT%type )

   STOP

99 CONTINUE
   WRITE( 6, "( ' array ', A, ' allocation failed' )" ) array_name
   STOP

   END PROGRAM GALAHAD_SLLS_EXAMPLE7
