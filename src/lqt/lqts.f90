   PROGRAM GALAHAD_LQT_EXAMPLE  !  GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.
   USE GALAHAD_LQT_DOUBLE                        ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 10000                ! problem dimension
   INTEGER :: i
   REAL ( KIND = working ) :: f, radius = 10.0_working  ! radius of ten
   REAL ( KIND = working ), DIMENSION( n ) :: X, C, H_vector
   TYPE ( LQT_data_type ) :: data
   TYPE ( LQT_control_type ) :: control
   TYPE ( LQT_inform_type ) :: inform
   CALL LQT_initialize( data, control, inform ) ! Initialize control parameters
   control%unitm = .FALSE.                ! M is not the identity matrix
   C = one                                ! The linear term is a vector of ones
   inform%status = 1
   control%print_level = 1
   DO                                     !  Iteration to find the minimizer
     CALL LQT_solve( n, radius, f, X, C, data, control, inform )
     SELECT CASE( inform%status )  ! Branch as a result of inform%status
     CASE( 2 )                  ! Form the preconditioned gradient
       data%U( : n ) = data%R( : n ) / two  ! Preconditioner is 2 times identity
     CASE ( 3 )                 ! Form the matrix-vector product
       data%Y( 1 ) = - two * data%Q( 1 ) + data%Q( 2 )
       DO i = 2, n - 1
         data%Y( i ) = data%Q( i - 1 ) - two * data%Q( i ) + data%Q( i + 1 )
       END DO
       data%Y( n ) = data%Q( n - 1 ) - two * data%Q( n )
     CASE ( - 30, 0 )  !  Successful return
       WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
      &    2ES12.4 )" ) inform%iter, f, inform%multiplier
       CALL LQT_terminate( data, control, inform ) ! delete internal workspace
        EXIT
     CASE DEFAULT      !  Error returns
       WRITE( 6, "( ' LQT_solve exit status = ', I6 ) " ) inform%status
       CALL LQT_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_LQT_EXAMPLE
