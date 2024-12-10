! THIS VERSION: GALAHAD 5.1 - 2024-11-23 AT 15:50 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_IR_TEST  !  further work needed!!
   USE GALAHAD_KINDS_precision
   USE GALAHAD_IR_precision                           ! double precision version
   USE GALAHAD_SMT_precision
   USE GALAHAD_SLS_precision
   IMPLICIT NONE
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_data_type ) :: SLS_data
   TYPE ( SLS_control_type ) SLS_control
   TYPE ( SLS_inform_type ) :: SLS_inform
   TYPE ( IR_data_type ) :: data
   TYPE ( IR_control_type ) :: control
   TYPE ( IR_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ne = 7
   REAL ( KIND = rp_ ) :: B( n ), X( n )
   INTEGER ( KIND = ip_ ) :: s
! Read matrix order and number of entries
   matrix%n = n
   matrix%ne = ne
! Allocate and set matrix
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 1, 2, 2, 3, 3, 5 /)
   matrix%col( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
   matrix%val( : ne ) = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 6.0_rp_, 1.0_rp_,        &
                           5.0_rp_, 1.0_rp_ /)
   CALL SMT_put( matrix%type, 'COORDINATE', s )     ! Specify co-ordinate
! Set right-hand side
   B( : n ) = (/ 8.0_rp_, 45.0_rp_, 31.0_rp_, 15.0_rp_, 17.0_rp_ /)
! Specify the solver (in this case sils)
   CALL SLS_initialize( 'sils', SLS_data, SLS_control, SLS_inform )
! Analyse
   CALL SLS_analyse( matrix, SLS_data, SLS_control, SLS_inform )
   IF ( SLS_inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_analyse with status = ', SLS_inform%status
     STOP
   END IF
! Factorize
   CALL SLS_factorize( matrix, SLS_data, SLS_control, SLS_inform )
   IF ( SLS_inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_factorize with status = ', SLS_inform%status
     STOP
   END IF
! solve using iterative refinement
   CALL IR_initialize( data, control, inform )    ! initialize IR structures
   control%itref_max = 2                          ! perform 2 iterations
   control%acceptable_residual_relative = 0.1 * EPSILON( 1.0D0 ) ! high accuracy
   X = B
   CALL IR_SOLVE( matrix, X, data, SLS_data, control, SLS_control, inform,     &
                  SLS_inform )
   IF ( inform%status == 0 ) THEN                 ! check for errors
     WRITE( 6, '( A, /, ( 5F10.6 ) )' ) ' Solution after refinement is', X
   ELSE
    WRITE( 6,'( A, I2 )' ) ' Failure of IR_solve with status = ', inform%status
   END IF
   CALL IR_terminate( data, control, inform )     ! delete internal workspace
   CALL SLS_terminate( SLS_data, SLS_control, SLS_inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM GALAHAD_IR_TEST
