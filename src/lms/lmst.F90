! THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 15:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LMS_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LMS_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, m = 3
   TYPE ( LMS_data_type ) :: data, data2
   TYPE ( LMS_control_type ) :: control, control2
   TYPE ( LMS_inform_type ) :: inform, inform2
   REAL ( KIND = rp_ ), DIMENSION( n ) :: S, Y, U, V
   INTEGER ( KIND = ip_ ) :: iter, fail, method, mem
   REAL ( KIND = rp_ ) :: delta, lambda
   TYPE ( RAND_seed ) :: seed
#ifdef REAL_32
   REAL ( KIND = sp_ ) :: error_tol = 1.0_sp_
#else
   REAL ( KIND = dp_ ) :: error_tol = 0.00001_dp_
#endif

! ==================== tests error returns ======================

   WRITE( 6, "( /, ' test error exits' )" )

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   CALL LMS_setup( 0_ip_, data, control, inform ) ! n <= 0
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   CALL LMS_setup( n, data, control, inform )
   delta = 0.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform ) ! delta <= 0
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   control%method = 4
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform, lambda = - 1.0_rp_ )
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   control%method = 4
   CALL LMS_setup( n, data, control, inform )
   delta = 0.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   CALL LMS_form_shift( - 1.0_rp_, data, control, inform ) ! lambda < 0
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = - S
   CALL LMS_form( S, Y, delta, data, control, inform ) ! s^T y <= 0
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   control%method = 3
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   CALL LMS_form_shift( 1.0_rp_, data, control, inform ) ! method = 3
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   CALL LMS_change_method( data, control, inform ) ! .not. any_method
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   CALL LMS_setup( n, data, control, inform )
   V = 1.0_rp_
   CALL LMS_apply( V, U, data, control, inform ) ! no call to form
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%print_level = 1
   control%method = 4
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   CALL LMS_apply( V, U, data, control, inform ) ! lambda not set
   WRITE( 6, "( ' exit status = ', I0 )" ) inform%status
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

! ==================== tests indididual methods ======================

   WRITE( 6, "( /, ' test individual methods' )" )

   CALL RAND_initialize( seed ) ! Initialize the random generator word
   DO method = 1, 5
     CALL LMS_initialize( data, control, inform ) !  initialize data
     control%memory_length = n + 2
     IF ( method == 5 ) THEN
       control%method = 4
     ELSE
       control%method = method
     END IF
     CALL LMS_setup( n, data, control, inform )
     fail = 0 ! count the failures
     DO iter = 1, 5 * n
       CALL RAND_random_real( seed, .FALSE., S )  ! pick random S, Y and delta
       CALL RAND_random_real( seed, .FALSE., Y )
       IF ( DOT_PRODUCT( S, Y ) < 0.0_rp_ ) Y = - Y
       CALL RAND_random_real( seed, .TRUE., delta )

       IF ( method == 5 ) THEN
         lambda = 0.0_rp_ ! apply the shifted L_BFGS (inverse) with zero shift
         CALL LMS_form( S, Y, delta, data, control, inform, lambda = lambda )
       ELSE
         CALL LMS_form( S, Y, delta, data, control, inform )
       END IF
       IF ( inform%status /= 0 ) THEN
         WRITE( 6, "( ' update error, status = ', I0 )" ) inform%status
         fail = fail + 1 ; CYCLE
       END IF

       IF ( method == 4 ) THEN
         lambda = 0.0_rp_ ! apply the shifted L_BFGS (inverse) with zero shift
         CALL LMS_form_shift( lambda, data, control, inform )
         IF ( inform2%status /= 0 ) THEN
           WRITE( 6, "( ' update error, status = ', I0 )" ) inform2%status
           fail = fail + 1 ; CYCLE
         END IF
       END IF

       V = 1.0_rp_
       CALL LMS_apply( V, U, data, control, inform ) ! form the required product
       IF ( inform%status /= 0 ) THEN
         WRITE( 6, "( ' apply error, status = ', I0 )" ) inform%status
         fail = fail + 1 ; CYCLE
       END IF
     END DO
     IF ( fail == 0 ) THEN  ! check for overall success
       WRITE( 6, "( ' method ', I0, ': no failures ' )" ) method
     ELSE
       WRITE( 6, "( ' method ', I0, ': ', I0, ' failures ' )" ) method, fail
     END IF
     CALL LMS_terminate( data, control, inform )  !  delete internal workspace
   END DO

! ============== generic tests with short and long_ memory =================

   DO mem = 1, 2
!  DO mem = 1, 1
     CALL LMS_initialize( data, control, inform ) !  initialize data

     IF ( mem == 1 ) THEN
       WRITE( 6, "( /, ' generic tests short memory' )" )
       control%memory_length = m
     ELSE
       WRITE( 6, "( /, ' generic tests long_ memory' )" )
       control%memory_length = n + 2
     END IF
     control2 = control
     control%method = 1
     control2%method = 4

     CALL LMS_setup( n, data, control, inform ) ! firstly apply L-BFGS
     CALL LMS_setup( n, data2, control2, inform2 ) ! then apply L_BFGS (inverse)

     fail = 0 ! count the failures
     DO iter = 1, 5 * n
!    DO iter = 1, 1
       CALL RAND_random_real( seed, .FALSE., S )  ! pick random S, Y and delta
       CALL RAND_random_real( seed, .FALSE., Y )
       IF ( DOT_PRODUCT( S, Y ) < 0.0_rp_ ) Y = - Y
       CALL RAND_random_real( seed, .TRUE., delta )

       CALL LMS_form( S, Y, delta, data, control, inform ) ! update the model
       IF ( inform%status /= 0 ) THEN
         WRITE( 6, "( ' form error, status = ', I0 )" ) inform%status
         fail = fail + 1 ; CYCLE
       END IF

       V = 1.0_rp_
       CALL LMS_apply( V, U, data, control, inform ) ! form the required product
       IF ( inform%status /= 0 ) THEN
         WRITE( 6, "( ' apply error, status = ', I0 )" ) inform%status
         fail = fail + 1 ; CYCLE
       END IF
       CALL LMS_form( S, Y, delta, data2, control2, inform2 ) ! update model 2
       IF ( inform2%status /= 0 ) THEN
         WRITE( 6, "( ' form error, status = ', I0 )" ) inform2%status
         fail = fail + 1 ; CYCLE
       END IF

       IF ( control2%method == 4 ) THEN
         lambda = 0.0_rp_ ! apply the shifted L_BFGS (inverse) with zero shift
         CALL LMS_form_shift( lambda, data2, control2, inform2 )
         IF ( inform2%status /= 0 ) THEN
           WRITE( 6, "( ' form_shift error, status = ', I0 )" ) inform2%status
           fail = fail + 1 ; CYCLE
         END IF
       END IF

       CALL LMS_apply( U, V, data2, control2, inform2 ) ! form the new product
       IF ( inform2%status /= 0 ) THEN
         WRITE( 6, "( ' apply error, status = ', I0 )" ) inform2%status
         fail = fail + 1 ; CYCLE
       END IF

       IF ( MAXVAL( ABS( V - 1.0_rp_ ) ) <= error_tol ) THEN ! check for error
!        WRITE( 6, "( ' iteration ', I2, ' error OK' )" ) iter
       ELSE
!        WRITE( 6, "( ' iteration ', I2, ' error too large' )" ) iter
         fail = fail + 1
       END IF
     END DO
     IF ( fail == 0 ) THEN  ! check for overall success
       WRITE( 6, "( ' no failures ' )" )
     ELSE
       WRITE( 6, "( 1X, I0, ' failures ' )" ) fail
     END IF
     CALL LMS_terminate( data, control, inform )  !  delete internal workspace
     CALL LMS_terminate( data2, control2, inform2 )
   END DO

! ==================== tests indididual methods ======================

   WRITE( 6, "( /, ' test changes of method' )" )

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%any_method = .TRUE. ; control%method = 1
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   fail = 0 ! count the failures
   DO method = 1, 4
     control%method = method
     IF ( method == 4 ) THEN
       CALL LMS_change_method( data, control, inform, lambda = 0.0_rp_ )
     ELSE
       CALL LMS_change_method( data, control, inform )
     END IF
     IF ( inform%status /= 0 ) THEN
       WRITE( 6, "( ' change_method error, status = ', I0 )" ) inform%status
       fail = fail + 1 ; CYCLE
     END IF
     V = 1.0_rp_
     CALL LMS_apply( V, U, data, control, inform ) ! form the required product
     IF ( inform%status /= 0 ) THEN
       WRITE( 6, "( ' apply error, status = ', I0 )" ) inform%status
       fail = fail + 1 ; CYCLE
     END IF
   END DO
   IF ( fail == 0 ) THEN  ! check for overall success
     WRITE( 6, "( ' no failures ' )" )
   ELSE
     WRITE( 6, "( 1X, I0, ' failures ' )" ) fail
   END IF
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

! ==================== tests sr1-skipping methods ======================

   WRITE( 6, "( /, ' test L-SR1 skips' )" )

   CALL LMS_initialize( data, control, inform ) !  initialize data
   control%method = 2
   CALL LMS_setup( n, data, control, inform )
   delta = 1.0_rp_ ; S = 1.0_rp_ ; Y = S
   CALL LMS_form( S, Y, delta, data, control, inform )
   IF ( inform%updates_skipped ) THEN  ! check for overall success
     WRITE( 6, "( ' an update was skipped as expected' )" )
   ELSE
     WRITE( 6, "( ' warning, no updates were skipped' )" )
   END IF
   V = 1.0_rp_
   CALL LMS_apply( V, U, data, control, inform ) ! form the required product
   IF ( inform%status /= 0 ) THEN
     WRITE( 6, "( ' apply error, status = ', I0 )" ) inform%status
   END IF
   CALL LMS_terminate( data, control, inform )  !  delete internal workspace

   END PROGRAM GALAHAD_LMS_test

