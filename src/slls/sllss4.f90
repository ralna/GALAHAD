! THIS VERSION: GALAHAD 4.1 - 2022-06-12 AT 10:15 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE4

! use slls to solve a bilinear fitting problem
!  min 1/2 sum_i ( xi y^T A_i x + b - mu_i)^2,
! where x and y lie in n and m regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with x free and y fixed, and vice versa

! For this example, n = 3m/2, A_i is given by subroutine calls (below)
! and mu_i is xi_* y_*^T A_i x_* + b_* for given x_*, y_*, xi_* and b_i*
   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_RAND_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p_x, p_y
   TYPE ( SLLS_data_type ) :: data_x, data_y
   TYPE ( SLLS_control_type ) :: control_x, control_y
   TYPE ( SLLS_inform_type ) :: inform_x, inform_y
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat, Y_stat
   INTEGER :: i, j, ne, pass
   TYPE ( RAND_seed ) :: seed
   INTEGER, PARAMETER :: eg = 2
   REAL ( KIND = wp ) :: val, f, xi, b, a_11, a_12, a_22, r_1, r_2
   INTEGER, PARAMETER :: m = 20, n = 3 * m / 2
   INTEGER, PARAMETER :: obs = 100
   INTEGER, PARAMETER :: pass_max = 3
   INTEGER, PARAMETER :: sol = 2
!  LOGICAL, PARAMETER :: pertub_observations = .TRUE.
   LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   REAL ( KIND = wp ) :: X( n ), Y( m ), Aix( m ), ATiy( n ), MU( obs )
   REAL ( KIND = wp ) :: AX( obs, n ), AY( obs, m )

! start problem data

! specify a solution

   SELECT CASE( sol )
! sparse solution
   CASE ( 1 )
!  solution is at x = 0 for i=1:n/2 and 2/n for i=n/2+1:n
     X( : n / 2 ) = 0.0_wp
     X( n / 2 + 1 : ) = 1.0_wp / REAL( n / 2, KIND = wp )
!  solution is at y = 2/m for i=1:m/2 and 0 for i=m/2+1:m
     Y( : m / 2 ) = 1.0_wp / REAL( m / 2, KIND = wp )
     Y( m / 2 + 1 : ) = 0.0_wp
! random dense solution
   CASE( 2 )
!  random x and y in (0,) scaled to satisfy e^T x = 1 = e^T y
     CALL RAND_initialize( seed )
     CALL RAND_random_real( seed, .TRUE., X( : n ) )
     val = SUM( X( : n ) ) ; X( : n ) = X( : n ) / val
     CALL RAND_random_real( seed, .TRUE., Y( : m ) )
     val = SUM( Y( : m ) ) ; Y( : m ) = Y( : m ) / val
   END SELECT
   xi = 1.0_wp ; b = 0.0_wp

!  set up storage for the x problem

   CALL SMT_put( p_x%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_x%A%val(obs * n ), p_x%B( obs ), p_x%X( n ), X_stat( n ) )
   p_x%m = obs ; p_x%n = n ; p_x%A%m = p_x%m ; p_x%A%n = p_x%n

!  set up storage for the y problem

   CALL SMT_put( p_y%A%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_y%A%val( obs * m ), p_y%B( obs ), p_y%X( m ), Y_stat( m ) )
   p_y%m = obs ; p_y%n = m ; p_y%A%m = p_y%m ; p_y%A%n = p_y%n

!  generate a set of observations, mu_i

   DO i = 1, obs
     CALL FORM_Aix( m, n, i, xi, X, Aix )
!    CALL FORM_ATiy( m, n, i, xi, Y, ATiy )
!write(6,"( ' X', /, ( 5ES12.4 ) )" ) X
!write(6,"( ' Aix', /, ( 5ES12.4 ) )" ) Aix
!write(6,"( ' Y', /, ( 5ES12.4 ) )" ) Y
!write(6,"( ' ATiy', /, ( 5ES12.4 ) )" ) ATiy
!    WRITE( 6, "( ' prods = ', 2ES12.4 )" ) &
!      DOT_PRODUCT( Y, Aix ), DOT_PRODUCT( X, ATiy )
     IF ( pertub_observations ) THEN
       CALL RAND_random_real( seed, .TRUE., val ) ; val = 0.001 * val
     ELSE
       val = 0.0_wp
     END IF
     MU( i ) = DOT_PRODUCT( Y, Aix ) + val
   END DO
write(6,"( ' MU', /, ( 5ES12.4 ) )" ) MU

!  starting point
   X = 1.0_wp / REAL( n, KIND = wp ) ; Y = 1.0_wp / REAL( m, KIND = wp )
   xi = 2.0_wp ; b = 1.0_wp

! problem data complete. Initialze data and control parameters

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

!  fix y, xi and b, and solve for x. Read in the rows of A

   ne = 0
   DO i = 1, obs
     CALL FORM_ATiy( m, n, i, xi, Y, p_x%A%val( ne + 1 : ne + n ) )
     ne = ne + n
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
     CALL FORM_Aix( m, n, i, xi, X, p_y%A%val( ne + 1 : ne + m ) )
     ne = ne + m
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
   ne = 0 ; a_11 = 0.0_wp ; a_12 = 0.0_wp ; r_1 = 0.0_wp
   DO i = 1, obs
     CALL FORM_Aix( m, n, i, 1.0_wp, X, p_y%A%val( ne + 1 : ne + m ) )
     val = DOT_PRODUCT( Y, p_y%A%val( ne + 1 : ne + m ) )
     ne = ne + n
     f = f + ( xi * val + b - MU( i ) ) ** 2
     a_11 = a_11 + val ** 2
     a_12 = a_12 + val
     r_1 = r_1 + val * MU( i )
   END DO
   WRITE( 6, "( /, ' 1/2|| F(x,y) - b||^2 = ', ES12.4 )" ) 0.5_wp * f
   a_22 = REAL( obs, KIND = WP )
   r_2 = SUM( MU( : obs ) )
   val = a_11 * a_22 - a_12 * a_12

   WRITE( 6, "( ' new mu, b = ', 2ES12.4 )" ) &
     ( a_22 * r_1 - a_12 * r_2 ) / val, &
     ( - a_12 * r_1 + a_11 * r_2 ) / val

   xi = ( a_22 * r_1 - a_12 * r_2 ) / val
   b = ( - a_12 * r_1 + a_11 * r_2 ) / val

!  report final residuals

   val = 0.0_wp
   DO i = 1, obs
     CALL FORM_Aix( m, n, i, xi, X, Aix )
     val = val + ( DOT_PRODUCT( Y, Aix ) + b - MU( i ) ) ** 2
   END DO
   WRITE( 6, "( /, ' 1/2|| F(x,y) - b||^2 = ', ES12.4 )" ) 0.5_wp * val

   pass = pass + 1
   IF ( pass <= pass_max .AND. SQRT( val ) > 0.00000001_wp ) GO TO 10

!  end of loop


! tidy up afterwards

   CALL SLLS_terminate( data_x, control_x, inform_x )  !  delete workspace
   DEALLOCATE( p_x%B, p_x%X, p_x%Z, X_stat )
   DEALLOCATE( p_x%A%val, p_x%A%type )
   CALL SLLS_terminate( data_y, control_y, inform_y )  !  delete workspace
   DEALLOCATE( p_y%B, p_y%X, p_y%Z, Y_stat )
   DEALLOCATE( p_y%A%val, p_y%A%type )


   CONTAINS
     SUBROUTINE FORM_Aix( m, n, i, xi, X, Aix )
     INTEGER :: m, n, i
     REAL ( KIND = wp ) :: xi
     REAL ( KIND = wp ) :: X( n )
     REAL ( KIND = wp ) :: Aix( m )
     REAL ( KIND = wp ) :: ri, partial
     ri = REAL( i , KIND = wp )
     SELECT CASE ( eg )

!        <-   m   -> <-n-m->
! A_i = ( 1 . 1 . 1 | i   0 )
!       (   .       |   .   )
!       ( 0   1   0 | 0   i ) <-n-m
!       (       .   .     . )
!       ( 0   0   1 | 1 . 1 )

     CASE( 1 )
       Aix( 1 ) = SUM( X( 1 : m ) )
       Aix( 2 : m - 1 ) = X( 2 : m - 1 )
       Aix( m ) = X( m ) + SUM( X( m + 1 : n ) )
       Aix( 1 : n - m ) = Aix( 1 : n - m ) + ri * X( m + 1 : n )

!         <-      m   ->   <-n-m->
! A_i = ( i+1 .  1   . 1  | 1   0 )
!       (     .      . .  |   .   )
!       (  1  . i+1  . 1  | 0   1 ) <-n-m
!       (  .  .      . .  |   . . )
!       (  1  .  1    i+1 | 0 . 0 )

     CASE( 2 )
       partial = SUM( X( 1 : m ) )
       Aix( : m ) = partial + ri * X( 1 : m )
       Aix( : n - m ) = Aix( : n - m ) + X( m + 1 : n )
     END SELECT
     Aix( : m ) = Aix( : m ) * xi
     END SUBROUTINE FORM_Aix
     SUBROUTINE FORM_ATiy( m, n, i, xi, Y, ATiy )
     INTEGER :: m, n, i
     REAL ( KIND = wp ) :: xi
     REAL ( KIND = wp ) :: Y( m )
     REAL ( KIND = wp ) :: ATiy( n )
     REAL ( KIND = wp ) :: ri, partial
     ri = REAL( i , KIND = wp )
     SELECT CASE ( eg )
     CASE( 1 )
       ATiy( 1 ) = Y( 1 )
       ATiy( 2 : m ) = Y( 1 ) + Y( 2 : m )
       ATiy( m + 1 : n ) = Y( m ) + ri * Y( 1 : n - m )
     CASE( 2 )
       partial = SUM( Y( : m ) )
       ATiy( 1 : m ) = partial + ri * Y( 1 : m )
       ATiy( m + 1 : n ) = Y( 1 : n - m )
     END SELECT
     ATiy( : n ) = ATiy( : n ) * xi
     END SUBROUTINE FORM_ATiy
   END PROGRAM GALAHAD_SLLS_EXAMPLE4
