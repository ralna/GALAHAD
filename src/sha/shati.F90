! THIS VERSION: GALAHAD 4.1 - 2023-08-20 AT 10:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SHA_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SHA_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   TYPE ( SHA_full_data_type ) :: data
   TYPE ( SHA_control_type ) :: control
   TYPE ( SHA_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: i, j, k, l, m, algorithm, status
   REAL ( KIND = rp_ ) ::  v
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
   TYPE ( RAND_seed ) :: seed
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, nz = 9 ! set problem data
   INTEGER ( KIND = ip_ ) :: ROW( nz ), COL( nz )
   REAL ( KIND = rp_ ) ::  VAL( nz ), VAL_est( nz )
   ROW = (/ 1, 1, 1, 1, 1, 2, 3, 4, 5 /)
   COL = (/ 1, 2, 3, 4, 5, 2, 3, 4, 5 /)
   VAL = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 /)
   DO algorithm = 1, 5
     CALL SHA_initialize( data, control, inform ) ! initialize
     control%approximation_algorithm = algorithm
     control%extra_differences = 1
     WRITE( 6, "( /, ' Approximation algorithm ', I0 )" ) algorithm
     CALL SHA_analyse_matrix( control, data, status, n, nz, ROW, COL, m )
     IF ( inform%status /= 0 ) THEN             ! Failure
       WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_analyse')") &
         inform%status ; STOP
     END IF
     WRITE( 6, "( 1X, I0, ' differences are needed, one extra might help')") m
     m = m + control%extra_differences ! use as many differences as required + 1
     ALLOCATE( S( n, m ), Y( n, m ), ORDER( m ) )
     CALL RAND_initialize( seed )
     DO k = 1, m
       DO i = 1, n                                             ! choose random S
         CALL RAND_random_real( seed, .FALSE., S( i, k ) )
       END DO
       Y( : n, k ) = 0.0_rp_                                    ! form Y = H * S
       DO l = 1, nz
         i = ROW( l ) ; j = COL( l ) ; v = VAL( l )
         Y( i, k ) = Y( i, k ) + v * S( j, k )
         IF ( i /= j ) Y( j, k ) = Y( j, k ) + v * S( i, k )
       END DO
       ORDER( k ) = m - k + 1  ! pick the (s,y) vectors in reverse order
     END DO
     ! approximate H
     CALL SHA_recover_matrix( data, status, m, S, Y, VAL_est,                  &
                              ORDER = ORDER )
     IF ( inform%status /= 0 ) THEN ! Failure
       WRITE( 6, "( ' return with nonzero status ',I0,' from SHA_estimate' )" )&
         inform%status
     ELSE
       WRITE( 6, "( /, ' Successful run with ', I0,                            &
      &             ' differences, estimated matrix:' )" ) m
       DO l = 1, nz
         WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES12.4, ')' )" ) &
          ROW( l ), COL( l ), VAL_est( l )
       END DO
     END IF
     CALL SHA_terminate( data, control, inform ) ! Delete internal workspace
     DEALLOCATE( S, Y, ORDER )
   END DO

   END PROGRAM GALAHAD_SHA_interface_test
