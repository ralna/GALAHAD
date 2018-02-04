   PROGRAM GALAHAD_L2RT_EXAMPLE   !  GALAHAD 2.4 - 15/05/20010 AT 14:15 GMT
   USE GALAHAD_L2RT_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )     ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n            ! problem dimensions
   INTEGER :: i
   REAL ( KIND = working ) :: p = 3.0_working         ! order of regulatisation
   REAL ( KIND = working ) :: sigma = 1.0_working     ! regulatisation weight
   REAL ( KIND = working ) :: mu = 0.0_working        ! zero shift
   REAL ( KIND = working ), DIMENSION( n ) :: X, V
   REAL ( KIND = working ), DIMENSION( m ) :: U, RES
   TYPE ( L2RT_data_type ) :: data
   TYPE ( L2RT_control_type ) :: control
   TYPE ( L2RT_inform_type ) :: inform
   CALL L2RT_initialize( data, control, inform ) ! Initialize control parameters
   control%fraction_opt = 0.99            ! Only require 99% of the best
   U = one                                ! The term b is a vector of ones
   inform%status = 1
   DO                                     ! Iteration to find the minimizer
      CALL L2RT_solve( m, n, p, sigma, mu, X, U, V, data, control, inform )
      SELECT CASE( inform%status )  !  Branch as a result of inform%status
      CASE( 2 )                     !  Form u <- u + A * v
        U( : n ) = U( : n ) + V     !  A^T = ( I : diag(1:n) )
        DO i = 1, n
          U( n + i ) = U( n + i ) + i * V( i )
        END DO
      CASE( 3 )                     !  Form v <- v + A^T * u
        V = V + U( : n )            
         DO i = 1, n
           V( i ) = V( i ) + i * U( n + i )
         END DO
      CASE ( 4 )                    !  Restart
         U = one                    !  re-initialize u to b
      CASE ( 0 )                    !  Successful return
         RES = one                  !  Compute the residuals for checking
         RES( : n )  = RES( : n ) - X
         DO i = 1, n
            RES( n + i ) = RES( n + i ) - i * X( i )
         END DO
         WRITE( 6, "( 1X, I0, ' 1st pass and ', I0, ' 2nd pass iterations' )" ) &
           inform%iter, inform%iter_pass2
         WRITE( 6, "( ' objective recurred and calculated = ', 2ES16.8 )" )     &
           inform%obj, SQRT( DOT_PRODUCT( RES, RES ) +                          &
             mu * SQRT( DOT_PRODUCT( X, X ) ) ** 2 ) + ( sigma / p ) *          &
           ( SQRT( DOT_PRODUCT( X, X ) ) ) ** p
         WRITE( 6, "( '    ||x||  recurred and calculated = ', 2ES16.8 )" )     &
           inform%x_norm, SQRT( DOT_PRODUCT( X, X ) )
         WRITE( 6, "( '  ||Ax-b|| recurred and calculated = ', 2ES16.8 )" )     &
           inform%r_norm, SQRT( DOT_PRODUCT( RES, RES ) )
         CALL L2RT_terminate( data, control, inform ) ! delete internal workspace
         EXIT
      CASE DEFAULT                  !  Error returns
         WRITE( 6, "( ' L2RT_solve exit status = ', I6 ) " ) inform%status
         CALL L2RT_terminate( data, control, inform ) ! delete internal workspace
         EXIT
      END SELECT
   END DO
   END PROGRAM GALAHAD_L2RT_EXAMPLE
