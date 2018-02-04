   PROGRAM GALAHAD_GLTR_EXAMPLE  !  GALAHAD 2.7 - 11/08/2016 AT 13:00 GMT.
   USE GALAHAD_GLTR_DOUBLE                        ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 10000                ! problem dimension
   INTEGER :: i
   REAL ( KIND = working ) :: f, radius = 10.0_working  ! radius of ten
   REAL ( KIND = working ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   TYPE ( GLTR_data_type ) :: data
   TYPE ( GLTR_control_type ) :: control
   TYPE ( GLTR_info_type ) :: inform
   CALL GLTR_initialize( data, control, inform ) ! Initialize control parameters
   control%unitm = .FALSE.                ! M is not the identity matrix
   R = one                                ! The linear term is a vector of ones
   inform%status = 1
   DO                                     !  Iteration to find the minimizer
     CALL GLTR_solve( n, radius, f, X, R, VECTOR, data, control, inform )
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
       CALL GLTR_terminate( data, control, inform ) ! delete internal workspace
        EXIT
     CASE DEFAULT      !  Error returns
       WRITE( 6, "( ' GLTR_solve exit status = ', I6 ) " ) inform%status
       CALL GLTR_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_GLTR_EXAMPLE
