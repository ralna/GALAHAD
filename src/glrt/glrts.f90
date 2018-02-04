   PROGRAM GALAHAD_GLRT_EXAMPLE  !  GALAHAD 2.7 - 08/02/2016 AT 09:50 GMT.
   USE GALAHAD_GLRT_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 )  ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 10000                 ! problem dimension
   INTEGER :: i
   REAL ( KIND = working ) :: p = 3.0_working      ! order of regulatisation
   REAL ( KIND = working ) :: eps = 1.0_working    ! shift
   REAL ( KIND = working ) :: sigma = 10.0_working ! regulatisation weight
   REAL ( KIND = working ), DIMENSION( n ) :: X, R, VECTOR, H_vector, O
   TYPE ( GLRT_data_type ) :: data
   TYPE ( GLRT_control_type ) :: control
   TYPE ( GLRT_inform_type ) :: inform
   CALL GLRT_initialize( data, control, inform ) ! Initialize control parameters
   control%unitm = .FALSE.              ! M is not the identity matrix
   control%fraction_opt = 0.99          ! Only require 99% of the best
   R = one                              ! The linear term c is a vector of ones
   O = - one                            ! The offset o is a vector of minus ones
   inform%status = 1
   DO                                    !  Iteration to find the minimizer
     CALL GLRT_solve( n, p, sigma, X, R, VECTOR, data, control, inform,        &
                      eps = eps, O = O )
     SELECT CASE( inform%status )       ! Branch as a result of inform%status
     CASE( 2 )                          ! Form the preconditioned vector
       VECTOR = VECTOR / two            ! Preconditioner is two times identity
     CASE ( 3 )                         ! Form the matrix-vector product
       H_vector( 1 ) = - two * VECTOR( 1 ) + VECTOR( 2 )
       DO i = 2, n - 1
         H_vector( i ) = VECTOR( i - 1 ) - two * VECTOR( i ) + VECTOR( i + 1 )
       END DO
       H_vector( n ) = VECTOR( n - 1 ) - two * VECTOR( n )
       VECTOR = H_vector
     CASE ( 4 )        !  Restart
       R = one         !  set r to c
     CASE( 5 )                          ! Form the product of the preconditioner
       VECTOR = two * VECTOR            ! with a vector
     CASE ( 0 )  !  Successful return
       WRITE( 6, "( 1X, I0, ' 1st pass and ', I0, ' 2nd pass iterations' )" )  &
         inform%iter, inform%iter_pass2
       H_vector( 1 ) = - two * X( 1 ) + X( 2 )
       DO i = 2, n - 1
         H_vector( i ) = X( i - 1 ) - two * X( i ) + X( i + 1 )
       END DO
       H_vector( n ) = X( n - 1 ) - two * X( n )
       WRITE( 6, "( ' objective recurred and calculated = ', 2ES16.8 )" )      &
        inform%obj_regularized, 0.5_working * DOT_PRODUCT( X, H_vector ) +     &
          SUM( X ) + ( sigma / p ) * ( two * DOT_PRODUCT( X + O, X + O ) +     &
            eps ) ** ( p/two )
       CALL GLRT_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     CASE DEFAULT      !  Error returns
       WRITE( 6, "( ' GLRT_solve exit status = ', I6 ) " ) inform%status
       CALL GLRT_terminate( data, control, inform ) ! delete internal workspace
       EXIT
     END SELECT
   END DO
   END PROGRAM GALAHAD_GLRT_EXAMPLE
