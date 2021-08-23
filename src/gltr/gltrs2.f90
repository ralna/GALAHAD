! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_GLTR_EXAMPLE2
   USE GALAHAD_GLTR_DOUBLE                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: working = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = working ), PARAMETER :: one = 1.0_working, two = 2.0_working
   INTEGER, PARAMETER :: n = 2                ! problem dimension
   INTEGER :: i
   REAL ( KIND = working ) :: f, radius = 1.0_working  ! radius of ten
   REAL ( KIND = working ), DIMENSION( n ) :: X, R, VECTOR, H_vector
   TYPE ( GLTR_data_type ) :: data
   TYPE ( GLTR_control_type ) :: control        
   TYPE ( GLTR_inform_type ) :: info
   CALL GLTR_initialize( data, control, info )  ! Initialize control parameters
   control%f_0 = 4.731884325266608D+0
!  control%print_level = 3
   control%out = 6
   control%unitm = .TRUE.                 ! M is the identity matrix
   R = (/ -4.637816414622023D+0, -1.222067920715109D-1 /)
   info%status = 1
   DO                                     !  Iteration to find the minimizer
      CALL GLTR_solve( n, radius, f, X, R, VECTOR, data, control, info )
      SELECT CASE( info%status )    ! Branch as a result of info%status
      CASE( 2, 6 )                  ! Form the preconditioned gradient
         VECTOR = VECTOR / two      ! Preconditioner is two times identity
      CASE ( 3, 7 )                 ! Form the matrix-vector product
        H_vector( 1 ) = 1.107272566595126D+3 * VECTOR( 1 ) + &
                        4.701123595505616D+2 * VECTOR( 2 )
        H_vector( 2 ) = 4.701123595505616D+2 * VECTOR( 1 ) + &
                        2.000000000000000D+2 * VECTOR( 2 )
        VECTOR = H_vector 
      CASE ( 5 )        !  Restart
         R = (/ -4.637816414622023D+0, -1.222067920715109D-1 /)
      CASE ( - 2 : 0 )  !  Successful return
         WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
        &       2ES12.4 )" ) info%iter + info%iter_pass2, f, info%multiplier
         CALL GLTR_terminate( data, control, info ) !  delete internal workspace
         EXIT
      CASE DEFAULT      !  Error returns
         WRITE( 6, "( ' GLTR_solve exit status = ', I6 ) " ) info%status
         CALL GLTR_terminate( data, control, info ) !  delete internal workspace
         EXIT
      END SELECT
   END DO
!  write(6,*) ' x ', X
   END PROGRAM GALAHAD_GLTR_EXAMPLE2
