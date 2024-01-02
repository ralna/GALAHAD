! THIS VERSION: GALAHAD 4.3 - 2023-12-31 AT 11:00 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE5

! use slls to solve a bilinear fitting problem
!  min 1/2 sum_i ( xi a^T_i x + b - mu_i)^2,
! where x lies in n regular simplex {x : e^T x = 1, x >= 0},
! using alternating solves with x free and (xi,b) fixed, and vice versa

   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_RAND_double
   USE GALAHAD_NORMS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p_r
   TYPE ( SLLS_data_type ) :: data_r
   TYPE ( SLLS_control_type ) :: control_r
   TYPE ( SLLS_inform_type ) :: inform_r
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: i, ne, pass
   REAL ( KIND = wp ) :: val, f, xi, b, a_max
   REAL ( KIND = wp ) :: a_11, a_12, a_22, r_1, r_2, alpha, beta
   REAL :: time, time_start, time_total
   INTEGER, PARAMETER :: n = 500
   INTEGER, PARAMETER :: obs = 200
   INTEGER, PARAMETER :: it_max = 100
   INTEGER, PARAMETER :: pass_max = 30
   INTEGER, DIMENSION( n ) :: R_stat
   REAL ( KIND = wp ) :: R( n ), AT( n, obs ), MU( obs ), SIGMA( obs )
   LOGICAL :: fileex

!  set up storage for the r problem

   CALL SMT_put( p_r%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_r%Ao%val(obs * n ), p_r%B( obs ), p_r%X( n ) )
   p_r%o = obs ; p_r%n = n ; p_r%Ao%m = p_r%o ; p_r%Ao%n = p_r%n

! read problem data. Store rows of A as columns of AT

   INQUIRE( file = "G.txt", exist = fileex )
   IF ( fileex ) THEN
     OPEN( unit = 21, file = "G.txt" )
   ELSE
     WRITE( 6, "( ' input file G.txt does not exist. Stopping' )" )
     STOP
   END IF
   DO i = 1, obs
     READ( 21, * ) AT( 1 : n, i )
   END DO
   CLOSE( unit = 21 )

   OPEN( unit = 22, file = "observation.txt" )
   a_max = 0.0_wp
   DO i = 1, obs
     READ( 22, * ) val, MU( i ), SIGMA( i )
     MU( i ) = MU( i ) / SIGMA( i )
     AT( 1 : n, i ) = AT( 1 : n, i ) / SIGMA( i )
     a_max = MAX ( a_max, TWO_NORM( AT( 1 : n, i ) ) )
!write(6,"( ' ||a||_2, b =', 2ES12.4 )" ) a_max, MU( i )
   END DO
   CLOSE( unit = 22 )

!  scale A and consequently xi

   AT = AT / a_max
!stop
!  starting point

   R = 1.0_wp / REAL( n, KIND = wp )
   xi = 1.0_wp ; b = 0.0_wp

!  report initial residuals

   f = 0.0_wp
   DO i = 1, obs
     alpha = DOT_PRODUCT( R( 1 : n ), AT( 1 : n, i ) )
     beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
   END DO
   WRITE( 6, "( /, ' obj =', ES22.14 )" ) 0.5_wp * f

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_r, control_r, inform_r )
   control_r%print_level = 1
!  control_r%exact_arc_search = .FALSE.
   control_r%exact_arc_search = .TRUE.
   control_r%maxit = it_max
!  control_r%convert_control%print_level = 3

!  pass

   CALL CPU_TIME( time_start )
   pass = 1
10 CONTINUE

   WRITE( 6, "( /, 1X, 32( '=' ), ' pass ', I0, 1X, 32( '=' ) )" ) pass

!  fix r and solve for xi and b. Read in the rows of A

   f = 0.0_wp
   a_11 = 0.0_wp ; a_12 = 0.0_wp ; a_22 = 0.0_wp ; r_1 = 0.0_wp ; r_2 = 0.0_wp
   DO i = 1, obs
     alpha = DOT_PRODUCT( R( 1 : n ), AT( 1 : n, i ) )
     beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
     a_11 = a_11 + alpha * alpha
     a_12 = a_12 + alpha * beta
     a_22 = a_22 + beta * beta
     r_1 = r_1 + alpha * MU( i )
     r_2 = r_2 + beta * MU( i )
   END DO
!  WRITE( 6, "(  ' a_11, a_12, a_22 ', 3ES12.4 )" ) a_11, a_12, a_22
   WRITE( 6, "( /, ' obj =', ES22.14 )" ) 0.5_wp * f
   val = a_11 * a_22 - a_12 * a_12

   WRITE( 6, "( ' new mu, b = ', 2ES12.4 )" ) &
     ( a_22 * r_1 - a_12 * r_2 ) / val, &
     ( - a_12 * r_1 + a_11 * r_2 ) / val

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  fix xi and b, and solve for r. Read in the rows of A

   ne = 0
   DO i = 1, obs
     p_r%B( i ) = MU( i ) - b / SIGMA( i )
     p_r%Ao%val( ne + 1 : ne + n ) = xi * AT( 1 : n, i )
     ne = ne + n
   END DO
   p_r%X = R

!  solve the problem

   IF ( pass >= pass_max ) control_r%maxit = 10000

   inform_r%status = 1
   CALL SLLS_solve( p_r, R_stat, data_r, control_r, inform_r, userdata )
   IF ( inform_r%status == 0 ) THEN           !  Successful return
     WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                        &
    &     ' Optimal objective value =', ES12.4 )" ) inform_r%iter, inform_r%obj
!    WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p_r%X
   ELSE                                       ! Error returns
     WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform_r%status
     WRITE( 6, * ) inform_r%alloc_status, inform_r%bad_alloc
   END IF

!  record the new r

   R = p_r%X

!  report final residuals

   f = 0.0_wp
   DO i = 1, obs
     alpha = DOT_PRODUCT( R( 1 : n ), AT( 1 : n, i ) )
     beta = 1.0_wp / SIGMA( i )
     f = f + ( alpha * xi + beta * b - MU( i ) ) ** 2
   END DO
   WRITE( 6, "( /, ' obj =', ES22.14 )" ) 0.5_wp * f
   WRITE( 6, "( ' xi, b = ', /, 2ES12.4 )" ) xi / a_max, b

   pass = pass + 1
   IF ( pass <= pass_max .AND. SQRT( f ) > 0.00000001_wp ) GO TO 10

!  end of loop

   WRITE( 6, "( /, 1X, 29( '=' ), ' pass limit ', 1X, 29( '=' ) )" )
   WRITE( 6, "( /, ' Optimal value =', ES22.14 )" ) 0.5_wp * f
!  WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) R
   WRITE( 6, "( ' Optimal xi, b = ', /, 2ES12.4 )" ) xi / a_max, b
   CALL CPU_TIME( time ) ; time_total = time - time_start
   WRITE( 6, "( ' total CPU time = ', F0.2, ' passes = ', I0 )" )              &
     time_total, pass

! tidy up afterwards

   CALL SLLS_terminate( data_r, control_r, inform_r )  !  delete workspace
   DEALLOCATE( p_r%B, p_r%X, p_r%Z )
   DEALLOCATE( p_r%Ao%val, p_r%Ao%type )

   END PROGRAM GALAHAD_SLLS_EXAMPLE5
