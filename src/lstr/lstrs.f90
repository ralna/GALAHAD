   PROGRAM GALAHAD_LSTR_EXAMPLE  !  GALAHAD 2.4 - 15/05/2010 AT 14:15 GMT
   USE GALAHAD_LSTR_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, zero = 0.0_working
   INTEGER, PARAMETER :: n = 50, m = 2 * n            ! problem dimensions
   INTEGER :: i
   REAL ( KIND = working ) :: radius = 1.0_working  ! radius of one
   REAL ( KIND = working ), DIMENSION( n ) :: X, V
   REAL ( KIND = working ), DIMENSION( m ) :: U, RES
   TYPE ( LSTR_data_type ) :: data
   TYPE ( LSTR_control_type ) :: control
   TYPE ( LSTR_inform_type ) :: inform
   CALL LSTR_initialize( data, control, inform ) ! Initialize control parameters
   control%steihaug_toint = .FALSE.       ! Try for an accurate solution
   control%fraction_opt = 0.99            ! Only require 99% of the best
   U = one                                ! The term b is a vector of ones
   inform%status = 1
   DO                                     ! Iteration to find the minimizer
     CALL LSTR_solve( m, n, radius, X, U, V, data, control, inform )
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
     CASE ( 4 )                   !  Restart
       U = one                    !  re-initialize u to b
     CASE ( - 30, 0 )             !  Successful return
       RES = one                  !  Compute the residuals for checking
       RES( : n )  = RES( : n ) - X
       DO i = 1, n
         RES( n + i ) = RES( n + i ) - i * X( i )
       END DO
       WRITE( 6, "( 1X, I0, ' 1st pass and ', I0, ' 2nd pass iterations' )" ) &
         inform%iter, inform%iter_pass2
       WRITE( 6, "( '   ||x||  recurred and calculated = ', 2ES16.8 )" )      &
         inform%x_norm, SQRT( DOT_PRODUCT( X, X ) )
       WRITE( 6, "( ' ||Ax-b|| recurred and calculated = ', 2ES16.8 )" )      &
         inform%r_norm, SQRT( DOT_PRODUCT( RES, RES ) )
       CALL LSTR_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     CASE DEFAULT                  !  Error returns
       WRITE( 6, "( ' LSTR_solve exit status = ', I6 ) " ) inform%status
       CALL LSTR_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_LSTR_EXAMPLE
