   PROGRAM GALAHAD_LQR_EXAMPLE  !  GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.
   USE GALAHAD_LQR_DOUBLE                        ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )     ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp, two = 2.0_wp
   INTEGER, PARAMETER :: n = 10000               ! problem dimension
   INTEGER :: i
   REAL ( KIND = wp ) :: f, radius = 10.0_wp  ! radius of ten
   REAL ( KIND = wp ) :: tau, omega, delta, kappa, eta, xi, m
   REAL ( KIND = wp ), DIMENSION( n ) :: X, C, G, V
   TYPE ( LQR_data_type ) :: data
   TYPE ( LQR_control_type ) :: control
   TYPE ( LQR_inform_type ) :: inform
   CALL LQR_initialize( data, control, inform ) ! Initialize control parameters
   control%unitm = .FALSE.                ! M is not the identity matrix
   C = one                                ! The linear term is a vector of ones
   inform%status = 1
   control%print_level = 1
!  control%stop_f_relative = 10.0_wp ** ( - 14 )
   DO                                     !  Iteration to find the minimizer
     CALL LQR_solve( n, radius, f, X, C, data, control, inform )
     SELECT CASE( inform%status )  ! Branch as a result of inform%status
     CASE( 2 )                  ! Form the preconditioned gradient
       data%U( : n ) = data%R( : n ) / two  ! Preconditioner is 2 times identity

       G = C
       G( 1 ) = G( 1 ) - two * X( 1 ) + X( 2 )
       DO i = 2, n - 1
         G( i ) = G( i ) + X( i - 1 ) - two * X( i ) + X( i + 1 )
       END DO
       G( n ) = G( n ) + X( n - 1 ) - two * X( n )
       G = G + two * data%lambda * X
!      WRITE( 6, "( ' g_norm^2 = ', ES22.14 )" )  DOT_PRODUCT( G, G ) / two
     CASE ( 3 )                 ! Form the matrix-vector product
       data%Y( 1 ) = - two * data%Q( 1 ) + data%Q( 2 )
       DO i = 2, n - 1
         data%Y( i ) = data%Q( i - 1 ) - two * data%Q( i ) + data%Q( i + 1 )
       END DO
       data%Y( n ) = data%Q( n - 1 ) - two * data%Q( n )
       IF ( .FALSE. .AND. inform%iter > 0 ) THEN
!      IF ( inform%iter > 0 ) THEN
         V( 1 ) = - two * X( 1 ) + X( 2 )
         DO i = 2, n - 1
           V( i ) = X( i - 1 ) - two * X( i ) + X( i + 1 )
         END DO
         V( n ) = X( n - 1 ) - two * X( n )
         tau = DOT_PRODUCT( V, X )
         omega = DOT_PRODUCT( V, data%Q( : n ) )
         eta = DOT_PRODUCT( C, V ) / two
         xi = DOT_PRODUCT( V, V ) / two
         kappa = DOT_PRODUCT( C, X )
         V( 1 ) = - two * data%Q( 1 ) + data%Q( 2 )
         DO i = 2, n - 1
           V( i ) = data%Q( i - 1 ) - two * data%Q( i ) + data%Q( i + 1 )
         END DO
         V( n ) = data%Q( n - 1 ) - two * data%Q( n )
         delta = DOT_PRODUCT( V, data%Q( : n ) )
         m = kappa + 0.5_wp * tau
write(6,"( ' calculated ' )" )
write(6,"( ' tau   = ', ES22.14, /, ' omega = ', ES22.14, /, &
&          ' delta = ', ES22.14, /, ' kappa = ', ES22.14, /, &
&          ' eta   = ', ES22.14, /, ' xi    = ', ES22.14, /, &
&          ' m     = ', ES22.14, / )" ) &
         tau, omega, delta, kappa, eta, xi, m
       END IF
     CASE ( 0, - 17 )  !  Successful return
       WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
      &    2ES12.4 )" ) inform%iter, f, inform%multiplier
       CALL LQR_terminate( data, control, inform ) ! delete internal workspace
        EXIT
     CASE DEFAULT      !  Error returns
       WRITE( 6, "( ' LQR_solve exit status = ', I6 ) " ) inform%status
       CALL LQR_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_LQR_EXAMPLE
