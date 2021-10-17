   PROGRAM GALAHAD_AQT_EXAMPLE  !  GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.
   USE GALAHAD_AQT_DOUBLE                        ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 10000                ! problem dimension
   INTEGER :: i
   REAL ( KIND = working ) :: f, radius = 10.0_working  ! radius of ten
   REAL ( KIND = working ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   TYPE ( AQT_data_type ) :: data
   TYPE ( AQT_control_type ) :: control
   TYPE ( AQT_inform_type ) :: inform
   CALL AQT_initialize( data, control, inform ) ! Initialize control parameters
   control%unitm = .FALSE.                ! M is not the identity matrix
   R = one                                ! The linear term is a vector of ones
   inform%status = 1
   DO                                     !  Iteration to find the minimizer
     CALL AQT_solve( n, radius, f, X, R, VECTOR, data, control, inform )
     SELECT CASE( inform%status )  ! Branch as a result of inform%status
     CASE( 2 )                  ! Form the preconditioned gradient
       VECTOR = VECTOR / two      ! Preconditioner is two times identity
     CASE ( 3 )                 ! Form the matrix-vector product
       H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
       DO i = 2, n - 1
         H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) + VECTOR( i + 1 )
       END DO
       H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
       VECTOR = H_vector
     CASE ( 5 )        !  Restart
       R = one
     CASE ( - 30, 0 )  !  Successful return
       WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
      &    2ES12.4 )" ) inform%iter + inform%iter_pass2, f, inform%multiplier
       CALL AQT_terminate( data, control, inform ) ! delete internal workspace
        EXIT
     CASE DEFAULT      !  Error returns
       WRITE( 6, "( ' AQT_solve exit status = ', I6 ) " ) inform%status
       CALL AQT_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_AQT_EXAMPLE
