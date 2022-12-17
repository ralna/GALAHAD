! THIS VERSION: GALAHAD 4.1 - 2022-12-17 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SHA_test_deck
   USE GALAHAD_PRECISION
   USE GALAHAD_SHA_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   TYPE ( SHA_data_type ) :: data
   TYPE ( SHA_control_type ) :: control        
   TYPE ( SHA_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: i, j, k, l, m
   REAL ( KIND = rp_ ) ::  v
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: RD
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
   TYPE ( RAND_seed ) :: seed
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, nz = 9        ! set problem data
   INTEGER ( KIND = ip_ ) :: ROW( nz ), COL( nz )
   REAL ( KIND = rp_ ) ::  VAL( nz ), VAL_est( nz )
!  ROW = (/ 1, 1, 1, 2, 2, 3, 3, 4, 5 /)
!  COL = (/ 1, 4, 5, 2, 4, 3, 5, 4, 5 /)
!  ROW = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5 /)
!  COL = (/ 1, 5, 2, 5, 3, 5, 4, 5, 5 /)
   ROW = (/ 1, 1, 1, 1, 1, 2, 3, 4, 5 /)
   COL = (/ 1, 2, 3, 4, 5, 2, 3, 4, 5 /)
   VAL = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 /)
   CALL SHA_initialize( data, control, inform )               ! initialize
   CALL SHA_analyse( n, nz, ROW, COL, data, control, inform ) ! analyse sparsity
   IF ( inform%status /= 0 ) THEN             ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_analyse' )" ) &
       inform%status ; STOP
   END IF
   m = inform%differences_needed
   ALLOCATE( S( n, m ), Y( n, m ), RD( m) )
   CALL RAND_initialize( seed )
   DO k = 1, m
     RD( k ) = k
     DO i = 1, n                                              ! choose random S
       CALL RAND_random_real( seed, .FALSE., S( i, k ) )
       CALL RAND_random_real( seed, .FALSE., Y( i, k ) )
       Y( i, k ) = Y( i, k ) * 0.001
     END DO   
     Y( : n, k ) = 0.0_rp_                                     ! form Y = H * S
     DO l = 1, nz
       i = ROW( l ) ; j = COL( l ) ; v = VAL( l )
       Y( i, k ) = Y( i, k ) + v * S( j, k )
       IF ( i /= j ) Y( j, k ) = Y( j, k ) + v * S( i, k )
     END DO
   END DO
   CALL SHA_estimate( n, nz, ROW, COL, m + 1, m, RD, n, m, S, n, m,            &
                      Y, VAL_est, data, control, inform )     ! approximate H
   IF ( inform%status /= 0 ) THEN             ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_estimate' )" )&
       inform%status ; STOP
   ELSE
     WRITE( 6, "( ' Successful run, estimated matrix:' )" )
     DO l = 1, nz
       WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES12.4, ')' )" )   &
        ROW( l ), COL( l ), VAL_est( l )
     END DO
   END IF
   WRITE( 6, "( ' TODO: test program is not yet exhausive' )" )
   CALL SHA_terminate( data, control, inform ) ! Delete internal workspace

   END PROGRAM GALAHAD_SHA_test_deck

