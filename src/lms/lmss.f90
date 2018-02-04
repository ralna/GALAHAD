! THIS VERSION: GALAHAD 2.6 - 12/06/2014 AT 15:30 GMT.
   PROGRAM GALAHAD_LMS_example
   USE GALAHAD_LMS_double                    ! double precision version
   USE GALAHAD_rand_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 5, m = 3
   TYPE ( LMS_data_type ) :: data, data2
   TYPE ( LMS_control_type ) :: control, control2
   TYPE ( LMS_inform_type ) :: inform, inform2
   REAL ( KIND = wp ), DIMENSION( n ) :: S, Y, U, V
   INTEGER :: iter, fail
   REAL ( KIND = wp ) :: delta, lambda
   TYPE ( RAND_seed ) :: seed
   CALL RAND_initialize( seed ) ! Initialize the random generator word
   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%memory_length = m ! set the memory length
   control2 = control
   control%method = 1 ! start with L-BFGS
   CALL LMS_setup( n, data, control, inform )
   control2%method = 3 ! then inverse L-BFGS
   control2%any_method =.TRUE. ! allow the 2nd update to change method
   CALL LMS_setup( n, data2, control2, inform2 )
   fail = 0 ! count the failures
   DO iter = 1, 5 * n
     IF ( iter == 3 * n ) THEN ! switch to inverse shifted L-BFGS
       CALL LMS_setup( n, data, control, inform )
       control2%method = 4
       CALL LMS_setup( n, data2, control2, inform2 )
     END IF
     CALL RAND_random_real( seed, .FALSE., S )  ! pick random S, Y and delta
     CALL RAND_random_real( seed, .FALSE., Y )
     IF ( DOT_PRODUCT( S, Y ) < 0.0_wp ) Y = - Y ! ensure that S^T Y is positive
     CALL RAND_random_real( seed, .TRUE., delta )
     CALL LMS_form( S, Y, delta, data, control, inform ) ! update the model
     IF ( inform%status /= 0 ) THEN
       WRITE( 6, "( ' update error, status = ', I0 )" ) inform%status
       fail = fail + 1 ; CYCLE
     END IF
     V = 1.0_wp ! form the first product with the vector ones
     CALL LMS_apply( V, U, data, control, inform ) ! form the required product
     IF ( inform%status /= 0 ) THEN
       WRITE( 6, "( ' apply error, status = ', I0 )" ) inform%status
       fail = fail + 1 ; CYCLE
     END IF
     CALL LMS_form( S, Y, delta, data2, control2, inform2 ) ! update model 2
     IF ( inform2%status /= 0 ) THEN
       WRITE( 6, "( ' update error, status = ', I0 )" ) inform2%status
       fail = fail + 1 ; CYCLE
     END IF
     IF ( control2%method == 4 ) THEN
       lambda = 0.0_wp ! apply the shifted L_BFGS (inverse) with zero shift
       CALL LMS_form_shift( lambda, data2, control2, inform2 )
       IF ( inform2%status /= 0 ) THEN
         WRITE( 6, "( ' update error, status = ', I0 )" ) inform2%status
         fail = fail + 1 ; CYCLE
       END IF
     END IF
! note, the preceeding two calls could have been condensed as
!    CALL LMS_form( S, Y, delta, data2, control2, inform2, lambda = 0.0_wp )
     CALL LMS_apply( U, V, data2, control2, inform2 ) ! form the new product
     IF ( inform2%status /= 0 ) THEN
       WRITE( 6, "( ' apply error, status = ', I0 )" ) inform2%status
       fail = fail + 1 ; CYCLE
     END IF
     IF ( MAXVAL( ABS( V - 1.0_wp ) ) > 0.00001_wp ) fail = fail + 1
   END DO
   IF ( fail == 0 ) THEN  ! check for overall success
     WRITE( 6, "( ' no failures ' )" )
   ELSE
     WRITE( 6, "( I0, ' failures ' )" ) fail
   END IF
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace
   CALL LMS_terminate( data2, control2, inform2 )
   END PROGRAM GALAHAD_LMS_example
