! THIS VERSION: GALAHAD 4.1 - 2022-06-12 AT 10:15 GMT
   PROGRAM GALAHAD_SLLS_EXAMPLE4

! use slls to solve a bilinear fitting problem 
!  min 1/2 sum_i ( y^T A_i x - b_i)^2,
! where x and y lie in n and m regular simplexes {z : e^T z = 1, z >= 0},
! using alternating solves with x free and y fixed, and vice versa

! For this example, n = 3m/2,
!
!        <-   m   -> <-n-m->
! A_i = ( 1 . 1 . 1 | i     )
!       (   .       |   .   )
!       (     1     |     i ) <-n-m
!       (       .   .     . )
!       (         1 | 1 . 1 )
!
! and b_i is y_*^T A_i x_* for given x_* and y_*

   USE GALAHAD_SLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p_x, p_y
   TYPE ( SLLS_data_type ) :: data_x, data_y
   TYPE ( SLLS_control_type ) :: control_x, control_y
   TYPE ( SLLS_inform_type ) :: inform_x, inform_y
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat, Y_stat
   INTEGER :: i, j, ne, s
   INTEGER, PARAMETER :: m = 20, n = 3 * m / 2
   INTEGER, PARAMETER :: obs = 100
   REAL ( KIND = wp ) :: X( n ), Y( m ), B( obs ), Aix( m ), ATiy( n )
   REAL ( KIND = wp ) :: AX( obs, n ), AY( obs, m )
! start problem data
!  solution is at x = 0 for i=1:n/2 and 2/n for i=n/2+1:n
   X( : n / 2 ) = 0.0_wp
   X( n / 2 + 1 : ) = 1.0_wp / REAL( n / 2, KIND = wp )
!  solution is at y = 2/m for i=1:m/2 and 0 for i=m/2+1:m 
   Y( : m / 2 ) = 1.0_wp / REAL( m / 2, KIND = wp )
   Y( m / 2 + 1 : ) = 0.0_wp

!  set up storage for the x problem

   CALL SMT_put( p_x%A%type, 'DENSE_BY_ROWS', s )
   ALLOCATE( p_x%A%val(obs * n ), p_x%B( obs ), p_x%X( n ), X_stat( n ) )
   p_x%m = obs ; p_x%n = n ; p_x%A%m = p_x%m ; p_x%A%n = p_x%n

!  set up storage for the y problem

   CALL SMT_put( p_y%A%type, 'DENSE_BY_ROWS', s )
   ALLOCATE( p_y%A%val( obs * m ), p_y%B( obs ), p_y%X( m ), Y_stat( m ) )
   p_y%m = obs ; p_y%n = m ; p_y%A%m = p_y%m ; p_y%A%n = p_y%n

   DO i = 1, obs
     CALL FORM_Aix( m, n, i, X, Aix )
!    CALL FORM_ATiy( m, n, i, Y, ATiy )
!write(6,"( ' X', /, ( 5ES12.4 ) )" ) X
!write(6,"( ' Aix', /, ( 5ES12.4 ) )" ) Aix
!write(6,"( ' Y', /, ( 5ES12.4 ) )" ) Y
!write(6,"( ' ATiy', /, ( 5ES12.4 ) )" ) ATiy
!    WRITE( 6, "( ' prods = ', 2ES12.4 )" ) &
!      DOT_PRODUCT( Y, Aix ), DOT_PRODUCT( X, ATiy )
     p_x%B( i ) = DOT_PRODUCT( Y, Aix ) ; p_y%B( i ) = p_x%B( i )
   END DO
!  starting point
   X = 1.0_wp / REAL( n, KIND = wp ) ; Y = 1.0_wp / REAL( m, KIND = wp )
!  DO i = 1, obs
!    CALL FORM_Aix( m, n, i, X, Aix )
!    WRITE( 6, "( ' r = ', ES12.4 )" ) DOT_PRODUCT( Y, Aix ) - B( i )
!  END DO
!  STOP

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

!  fix y and solve for x. Read in the rows of A

   ne = 0
   DO i = 1, obs
     CALL FORM_ATiy( m, n, i, Y, p_x%A%val( ne + 1 : ne + n ) )
     ne = ne + n
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

!  fix x and solve for y. Read in the rows of A

   ne = 0
   DO i = 1, obs
     CALL FORM_Aix( m, n, i, X, p_y%A%val( ne + 1 : ne + m ) )
     ne = ne + m
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

!  report final residuals

   DO i = 1, obs
     CALL FORM_Aix( m, n, i, X, Aix )
     WRITE( 6, "( ' r = ', ES12.4 )" ) DOT_PRODUCT( Y, Aix ) - p_x%B( i )
   END DO



! tidy up afterwards

   CALL SLLS_terminate( data_x, control_x, inform_x )  !  delete workspace
   DEALLOCATE( p_x%B, p_x%X, p_x%Z, X_stat )
   DEALLOCATE( p_x%A%val, p_x%A%type )
   CALL SLLS_terminate( data_y, control_y, inform_y )  !  delete workspace
   DEALLOCATE( p_y%B, p_y%X, p_y%Z, Y_stat )
   DEALLOCATE( p_y%A%val, p_y%A%type )


   CONTAINS
     SUBROUTINE FORM_Aix( m, n, i, X, Aix )
     INTEGER :: m, n, i
     REAL ( KIND = wp ) :: X( n )
     REAL ( KIND = wp ) :: Aix( m )
     REAL ( KIND = wp ) :: ri
     ri = REAL( i , KIND = wp )
     Aix( 1 ) = SUM( X( : m ) )
     Aix( 2 : m - 1 ) = X( 2 : m - 1 )
     Aix( m ) = X( m ) + SUM( X( m + 1 : n ) )
     Aix( 1 : n - m ) = Aix( 1 : n - m ) + ri * X( m + 1 : n )
     END SUBROUTINE FORM_Aix
     SUBROUTINE FORM_ATiy( m, n, i, Y, ATiy )
     INTEGER :: m, n, i
     REAL ( KIND = wp ) :: Y( m )
     REAL ( KIND = wp ) :: ATiy( n )
     REAL ( KIND = wp ) :: ri
     ri = REAL( i , KIND = wp )
     ATiy( 1 ) = Y( 1 )
     ATiy( 2 : m ) = Y( 1 ) + Y( 2 : m )
     ATiy( m + 1 : n ) = Y( m ) + ri * Y( 1 : n - m )
     END SUBROUTINE FORM_ATiy
   END PROGRAM GALAHAD_SLLS_EXAMPLE4

