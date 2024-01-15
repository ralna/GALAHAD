#include "galahad_modules.h"
   PROGRAM ICFS_TEST_PROGRAM   !  GALAHAD 4.1 - 2022-12-14 AT 10:00 GMT.
   USE GALAHAD_KINDS_precision
   USE GALAHAD_ICFS_precision
   IMPLICIT NONE
   TYPE ( ICFS_data_type ) :: data
   TYPE ( ICFS_control_type ) control
   TYPE ( ICFS_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ne = 5
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PTR, ROW
   INTEGER ( KIND = ip_ ) :: i
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : )  :: DIAG, VAL
   REAL ( KIND = rp_ ) :: X( n )
! allocate and set lower triangle of matrix in sparse by column form
   ALLOCATE( PTR( n + 1 ), DIAG( n ), VAL( ne ), ROW( ne ) )
   PTR = (/ 1, 2, 4, 5, 5, 5 /)
   ROW = (/ 2, 3, 5, 4 /)
   DIAG = (/ 2.0_rp_,  5.0_rp_,  1.0_rp_,  7.0_rp_,  2.0_rp_ /)
   VAL = (/ 1.0_rp_,  1.0_rp_,  1.0_rp_,  1.0_rp_ /)
! problem setup complete
   CALL ICFS_initialize( data, control, inform )
   DO i = 0, 2
     control%icfs_vectors = i
! form and factorize the preconditioner, P = L L^T
     CALL ICFS_factorize( n, PTR, ROW, DIAG, VAL, data, control, inform )
     IF ( inform%status < 0 ) THEN
       WRITE( 6, '( A, I0 )' )                                                 &
            ' Failure of ICFS_factorize with status = ', inform%status
       STOP
     END IF
! use the factors to solve L L^T x = b, with b input in x
     X( : n ) = (/ 3.0_rp_,  8.0_rp_,  3.0_rp_,  8.0_rp_,  3.0_rp_ /)
     CALL ICFS_triangular_solve( n, X, .FALSE., data, control, inform )
     CALL ICFS_triangular_solve( n, X, .TRUE., data, control, inform )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( ' ICFS - Preconditioned solution is ', 5F6.2 )" ) X
     ELSE
       WRITE( 6, "( ' ICFS - exit status = ', I0 )" ) inform%status
     END IF
   END DO
! clean up
   CALL ICFS_terminate( data, control, inform )
   DEALLOCATE( DIAG, VAL, ROW, PTR )
   STOP
   END PROGRAM ICFS_TEST_PROGRAM
