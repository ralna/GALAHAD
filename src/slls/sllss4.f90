! THIS VERSION: GALAHAD 4.3 - 2023-12-31 AT 10:00 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE4

! use slls to solve a bilinear fitting problem
!  min 1/2 sum_i ( xi y^T A_i x + b - mu_i)^2,
! where x and y lie in n and o regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with x free and y fixed, and vice versa

! For this example, n = 3o/2, A_i is given by subroutine calls (below)
! and mu_i is xi_* y_*^T A_i x_* + b_* for given x_*, y_*, xi_* and b_i*
   USE GALAHAD_SLLS_double         ! double precision version
   USE GALAHAD_RAND_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( QPT_problem_type ) :: p_x, p_y
   TYPE ( SLLS_data_type ) :: data_x, data_y
   TYPE ( SLLS_control_type ) :: control_x, control_y
   TYPE ( SLLS_inform_type ) :: inform_x, inform_y
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat, Y_stat
   INTEGER :: i, ne, pass
   TYPE ( RAND_seed ) :: seed
   INTEGER, PARAMETER :: eg = 2
   REAL ( KIND = wp ) :: val, f, xi, b, a_11, a_12, a_22, r_1, r_2
   INTEGER, PARAMETER :: o = 20, n = 3 * o / 2
   INTEGER, PARAMETER :: obs = 100
   INTEGER, PARAMETER :: pass_max = 3
   INTEGER, PARAMETER :: sol = 2
!  LOGICAL, PARAMETER :: pertub_observations = .TRUE.
   LOGICAL, PARAMETER :: pertub_observations = .FALSE.
   REAL ( KIND = wp ) :: X( n ), Y( o ), Aix( o ), MU( obs )
!  REAL ( KIND = wp ) :: ATiy( n ), AY( obs, o )
!  REAL ( KIND = wp ) :: AX( obs, n )

! start problem data

! specify a solution

   SELECT CASE( sol )
! sparse solution
   CASE ( 1 )
!  solution is at x = 0 for i=1:n/2 and 2/n for i=n/2+1:n
     X( : n / 2 ) = 0.0_wp
     X( n / 2 + 1 : ) = 1.0_wp / REAL( n / 2, KIND = wp )
!  solution is at y = 2/m for i=1:m/2 and 0 for i=m/2+1:m
     Y( : o / 2 ) = 1.0_wp / REAL( o / 2, KIND = wp )
     Y( o / 2 + 1 : ) = 0.0_wp
! random dense solution
   CASE( 2 )
!  random x and y in (0,) scaled to satisfy e^T x = 1 = e^T y
     CALL RAND_initialize( seed )
     CALL RAND_random_real( seed, .TRUE., X( : n ) )
     val = SUM( X( : n ) ) ; X( : n ) = X( : n ) / val
     CALL RAND_random_real( seed, .TRUE., Y( : o ) )
     val = SUM( Y( : o ) ) ; Y( : o ) = Y( : o ) / val
   END SELECT
   xi = 1.0_wp ; b = 0.0_wp

!  set up storage for the x problem

   CALL SMT_put( p_x%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_x%Ao%val(obs * n ), p_x%B( obs ), p_x%X( n ), X_stat( n ) )
   p_x%o = obs ; p_x%n = n ; p_x%Ao%m = p_x%o ; p_x%Ao%n = p_x%n

!  set up storage for the y problem

   CALL SMT_put( p_y%Ao%type, 'DENSE_BY_ROWS', i )
   ALLOCATE( p_y%Ao%val( obs * o ), p_y%B( obs ), p_y%X( o ), Y_stat( o ) )
   p_y%o = obs ; p_y%n = o ; p_y%Ao%m = p_y%o ; p_y%Ao%n = p_y%n

!  generate a set of observations, mu_i

   DO i = 1, obs
     CALL FORM_Aix( o, n, i, xi, X, Aix )
!    CALL FORM_ATiy( o, n, i, xi, Y, ATiy )
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
   X = 1.0_wp / REAL( n, KIND = wp ) ; Y = 1.0_wp / REAL( o, KIND = wp )
   xi = 2.0_wp ; b = 1.0_wp

! problem data complete. Initialze data and control parameters

   CALL SLLS_initialize( data_x, control_x, inform_x )
   control_x%print_level = 1
   control_x%exact_arc_search = .FALSE.
!  control_x%convert_control%print_level = 3

   CALL SLLS_initialize( data_y, control_y, inform_y )
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
     CALL FORM_ATiy( o, n, i, xi, Y, p_x%Ao%val( ne + 1 : ne + n ) )
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
     CALL FORM_Aix( o, n, i, xi, X, p_y%Ao%val( ne + 1 : ne + o ) )
     ne = ne + o
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
     CALL FORM_Aix( o, n, i, 1.0_wp, X, p_y%Ao%val( ne + 1 : ne + o ) )
     val = DOT_PRODUCT( Y, p_y%Ao%val( ne + 1 : ne + o ) )
     ne = ne + o
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
     CALL FORM_Aix( o, n, i, xi, X, Aix )
     val = val + ( DOT_PRODUCT( Y, Aix ) + b - MU( i ) ) ** 2
   END DO
   WRITE( 6, "( /, ' 1/2|| F(x,y) - b||^2 = ', ES12.4 )" ) 0.5_wp * val

   pass = pass + 1
   IF ( pass <= pass_max .AND. SQRT( val ) > 0.00000001_wp ) GO TO 10

!  end of loop


! tidy up afterwards

   CALL SLLS_terminate( data_x, control_x, inform_x )  !  delete workspace
   DEALLOCATE( p_x%B, p_x%X, p_x%Z, X_stat )
   DEALLOCATE( p_x%Ao%val, p_x%Ao%type )
   CALL SLLS_terminate( data_y, control_y, inform_y )  !  delete workspace
   DEALLOCATE( p_y%B, p_y%X, p_y%Z, Y_stat )
   DEALLOCATE( p_y%Ao%val, p_y%Ao%type )


   CONTAINS
     SUBROUTINE FORM_Aix( o, n, i, xi, X, Aix )
     INTEGER :: o, n, i
     REAL ( KIND = wp ) :: xi
     REAL ( KIND = wp ) :: X( n )
     REAL ( KIND = wp ) :: Aix( o )
     REAL ( KIND = wp ) :: ri, partial
     ri = REAL( i , KIND = wp )
     SELECT CASE ( eg )

!        <-   o   -> <-n-m->
! A_i = ( 1 . 1 . 1 | i   0 )
!       (   .       |   .   )
!       ( 0   1   0 | 0   i ) <-n-m
!       (       .   .     . )
!       ( 0   0   1 | 1 . 1 )

     CASE( 1 )
       Aix( 1 ) = SUM( X( 1 : o ) )
       Aix( 2 : o - 1 ) = X( 2 : o - 1 )
       Aix( o ) = X( o ) + SUM( X( o + 1 : n ) )
       Aix( 1 : n - o ) = Aix( 1 : n - o ) + ri * X( o + 1 : n )

!         <-      o   ->   <-n-m->
! A_i = ( i+1 .  1   . 1  | 1   0 )
!       (     .      . .  |   .   )
!       (  1  . i+1  . 1  | 0   1 ) <-n-m
!       (  .  .      . .  |   . . )
!       (  1  .  1    i+1 | 0 . 0 )

     CASE( 2 )
       partial = SUM( X( 1 : o ) )
       Aix( : o ) = partial + ri * X( 1 : o )
       Aix( : n - o ) = Aix( : n - o ) + X( o + 1 : n )
     END SELECT
     Aix( : o ) = Aix( : o ) * xi
     END SUBROUTINE FORM_Aix
     SUBROUTINE FORM_ATiy( o, n, i, xi, Y, ATiy )
     INTEGER :: o, n, i
     REAL ( KIND = wp ) :: xi
     REAL ( KIND = wp ) :: Y( o )
     REAL ( KIND = wp ) :: ATiy( n )
     REAL ( KIND = wp ) :: ri, partial
     ri = REAL( i , KIND = wp )
     SELECT CASE ( eg )
     CASE( 1 )
       ATiy( 1 ) = Y( 1 )
       ATiy( 2 : o ) = Y( 1 ) + Y( 2 : o )
       ATiy( o + 1 : n ) = Y( o ) + ri * Y( 1 : n - o )
     CASE( 2 )
       partial = SUM( Y( : o ) )
       ATiy( 1 : o ) = partial + ri * Y( 1 : o )
       ATiy( o + 1 : n ) = Y( 1 : n - o )
     END SELECT
     ATiy( : n ) = ATiy( : n ) * xi
     END SUBROUTINE FORM_ATiy
   END PROGRAM GALAHAD_SLLS_EXAMPLE4
