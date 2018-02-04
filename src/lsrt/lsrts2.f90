! THIS VERSION: GALAHAD 2.2 - 26/05/2008 AT 10:30 GMT.
   PROGRAM GALAHAD_LSRT2_BIT_TEST
   USE GALAHAD_LSRT_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: dim_max = 3
   INTEGER, PARAMETER :: cond_max = 2
   INTEGER, PARAMETER :: sig_max = 5
   INTEGER :: it_min( sig_max, cond_max, dim_max )
   INTEGER :: it_max( sig_max, cond_max, dim_max )
   REAL ( KIND = working ) :: p = 3.0_working         ! order of regulatisation
   REAL ( KIND = working ) :: it_mean( sig_max, cond_max, dim_max )
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   REAL ( KIND = working ), PARAMETER :: two = 2.0_working
   INTEGER :: m, n, i, mn, dim, sig, cond
   REAL ( KIND = working ) :: sigma, rho, c, e, tyty, tztz, prod
   REAL ( KIND = working ) :: sigmae( sig_max )
   REAL ( KIND = working ) :: rhos( cond_max )
   REAL ( KIND = working ), ALLOCATABLE, DIMENSION( : ) :: X, V, U, RES, Y, Z, D
   REAL ( KIND = working ), ALLOCATABLE, DIMENSION( : ) :: Wm, Wn
   TYPE ( LSRT_data_type ) :: data
   TYPE ( LSRT_control_type ) :: control
   TYPE ( LSRT_inform_type ) :: inform
   sigmae = (/ 0.0001_working, 0.01_working, 1.0_working, 100.0_working,        &
               10000.0_working /)
   rhos = (/ 0.01_working, 0.0001_working /)

!  Set up data -

!  A = ( I_m - yy^T/2y^Ty) D ( I_n - zz^T/2z^Tz), where
!  y_i = 1 (i=1:n), z_i = 1 (i=1:n:2), -1 (i=2:n:2) and 
!  D=0 except D_ii = c + e*i (i=1:min(m,n)), 
!  e = (1-rho)/(1-min(m,n)), c = 1-e
!  b_i = 1

   WRITE( 44, "( '   m     n       cond       sigma    min mean max ' )" )

   it_min = 0 ; it_max = 0 ; it_mean = zero
   DO dim = 1, dim_max
!  DO dim = 3, 3
    SELECT CASE( dim )
    CASE ( 1 ) 
      n = 5000
      m = 1000
    CASE ( 2 ) 
      n = 1000
      m = 5000
    CASE ( 3 ) 
      n = 5000
      m = 5000
    END SELECT

    mn = MIN( m, n )
    ALLOCATE( X( n ), V( n ), U( m ), RES( m ), Y( m ), Z( n ), D( mn ) )
    ALLOCATE( Wm( m ), Wn( n ) )
    Y = one
    DO i = 1, n, 2
      Z( i ) = one ; Z( i + 1 ) = - one
    END DO 
    tyty = two * DOT_PRODUCT( Y, Y )
    tztz = two * DOT_PRODUCT( Z, Z )

    DO cond = 1, cond_max
!   DO cond = 2, 2
     rho = rhos( cond )
 
     e = ( one - rho ) / ( one - mn ) ; c = one - e
     DO i = 1, mn
       D( i ) = c + e * i
     END DO
   
     DO sig = 1, sig_max
!    DO sig = 3, 3
      sigma = sigmae( sig )

      CALL LSRT_initialize( data, control, inform ) ! Initialize control parameters
      control%fraction_opt = 0.99            ! Only require 99% of the best
!     control%print_level = 2
!     control%itmax = 4761

      U = one                 ! The term b
!     U( m ) = one ; U( 1 : m - 1 ) = zero                 ! The term b
      inform%status = 1
      DO  ! Iteration to find the minimizer
        CALL LSRT_solve( m, n, p, sigma, X, U, V, data, control, inform )
        SELECT CASE( inform%status )  !  Branch as a result of inform%status
        CASE( 2 )                     !  Form u <- u + A * v
          prod = DOT_PRODUCT( V, Z ) / tztz
          Wn = V - prod * Z
          IF ( m <= n ) THEN
            Wm = Wn( : m ) * D( : m )
          ELSE
            Wm( : n ) = Wn * D
            Wm( n + 1 : ) = zero
          END IF
          prod = DOT_PRODUCT( Wm, Y ) / tyty
          U = U + Wm - prod * Y
        CASE( 3 )                     !  Form v <- v + A^T * u
          prod = DOT_PRODUCT( U, Y ) / tyty
          Wm = U - prod * Y
          IF ( n <= m ) THEN
            Wn = Wm( : n ) * D( : n )
          ELSE
            Wn( : m ) = Wm * D
            Wn( m + 1 : ) = zero
          END IF
          prod = DOT_PRODUCT( Wn, Z ) / tztz
          V = V + Wn - prod * Z
        CASE ( 4 )                    !  Restart
!         U( m ) = one ; U( 1 : m - 1 ) = zero ! re-initialize u to b
          U = one                     !  re-initialize u to b
        CASE ( - 2 : 0 )              !  Successful return
          prod = DOT_PRODUCT( X, Z ) / tztz
          Wn = X - prod * Z
          IF ( m <= n ) THEN
            Wm = Wn( : m ) * D( : m )
          ELSE
            Wm( : n ) = Wn * D
            Wm( n + 1 : ) = zero
          END IF
          prod = DOT_PRODUCT( Wm, Y ) / tyty
          RES = Wm - prod * Y - one
          WRITE( 6, "( 1X, I0, ' 1st pass and ', I0, ' 2nd pass iterations' )" )&
            inform%iter, inform%iter_pass2
          WRITE( 6, "( ' objective recurred and calculated = ', 2ES16.8 )" )    &
            inform%obj, 0.5_working * DOT_PRODUCT( RES, RES ) + ( sigma / p ) * &
            ( SQRT( DOT_PRODUCT( X, X ) ) ) ** p
          WRITE( 6, "( '   ||x||  recurred and calculated = ', 2ES16.8 )" )     &
            inform%x_norm, SQRT( DOT_PRODUCT( X, X ) )
          WRITE( 6, "( ' ||Ax-b|| recurred and calculated = ', 2ES16.8 )" )     &
            inform%r_norm, SQRT( DOT_PRODUCT( RES, RES ) )
          WRITE( 6, "( ' m, n, cond, sigma ',  I0, 1X, I0, 2ES12.4 )" )        &
            m, n, one / rho, sigma
          WRITE( 6, "( ' min, mean, max, number boundary iterations ',          &
         & I0, F4.1, 1X, I0, 1X, I0 )" ) inform%biter_min, inform%biter_mean,   &
             inform%biter_max, inform%biters
          WRITE( 44, "( 2I6, 2ES12.4, I4, F5.1, I4, I6 )" ) m, n, one / rho,    &
            sigma, inform%biter_min, inform%biter_mean,                        &
             inform%biter_max, inform%biters
          CALL LSRT_terminate( data, control, inform ) !delete internal workspace
          it_min( sig, cond, dim ) = inform%biter_min
          it_max( sig, cond, dim ) = inform%biter_max
          it_mean( sig, cond, dim ) = inform%biter_mean
          EXIT
        CASE DEFAULT !  Error returns
          WRITE( 6, "( ' LSRT_solve exit status = ', I6 ) " ) inform%status
          CALL LSRT_terminate( data, control, inform ) !delete internal workspace
          EXIT
        END SELECT
      END DO
     END DO
    END DO
    DEALLOCATE( X, V, U, RES, Y, Z, D, Wm, Wn )
   END DO

   DO sig = 1, sig_max
    DO cond = 1, cond_max 
      WRITE( 45, "( F0.5, ' & ' , F0.0, ' & ' )" )                             &
         sigmae( sig ), one / rhos( cond )
     WRITE( 45, "( 3 ( I4, ' & ',  F5.1, ' & ', I4, ' & ' ) )" )               &
       ( it_min( sig,  cond, dim ), it_mean( sig,  cond, dim ),                &
         it_max( sig,  cond, dim ), dim = 1, dim_max )
   END DO
   END DO

   END PROGRAM GALAHAD_LSRT2_BIT_TEST
