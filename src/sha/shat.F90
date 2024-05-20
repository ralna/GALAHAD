! THIS VERSION: GALAHAD 4.1 - 2023-08-19 AT 15:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SHA_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SHA_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   TYPE ( SHA_data_type ) :: data
   TYPE ( SHA_control_type ) :: control
   TYPE ( SHA_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: i, j, k, l, m, algorithm
   REAL ( KIND = rp_ ) ::  v
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER
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
   DO algorithm = 1, 8
     CALL SHA_initialize( data, control, inform )              ! initialize
     control%approximation_algorithm = algorithm
     control%sparse_row = 3
     WRITE( 6, "( /, ' Approximation algorithm ', I0 )" ) algorithm
     CALL SHA_analyse( n, nz, ROW, COL, data, control, inform )!analyse sparsity
     IF ( inform%status /= 0 ) THEN             ! Failure
       WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_analyse')") &
         inform%status ; STOP
     END IF
     WRITE( 6, "( 1X, I0, ' differences are needed, one extra might help')")   &
       inform%differences_needed
     m = inform%differences_needed + 1 ! use as many differences as required + 1
     ALLOCATE( S( n, m ), Y( n, m ), ORDER( m ) )
     CALL RAND_initialize( seed )
     DO k = 1, m
       ORDER( k ) = m - k + 1
       DO i = 1, n  ! choose random S
         CALL RAND_random_real( seed, .FALSE., S( i, k ) )
       END DO
       Y( : n, k ) = 0.0_rp_  ! form Y = H * S
       DO l = 1, nz
         i = ROW( l ) ; j = COL( l ) ; v = VAL( l )
         Y( i, k ) = Y( i, k ) + v * S( j, k )
         IF ( i /= j ) Y( j, k ) = Y( j, k ) + v * S( i, k )
       END DO
     END DO
     ! approximate H
     CALL SHA_estimate( n, nz, ROW, COL, m, S, n, m, Y, n, m, VAL_est,         &
                        data, control, inform )
     IF ( inform%status /= 0 ) THEN ! Failure
       WRITE( 6, "( ' return with nonzero status ',I0,' from SHA_estimate' )" )&
         inform%status
     ELSE
!      WRITE( 6, "( /, ' Successful run with ', I0,                            &
!     &             ' differences, estimated matrix:' )" ) m
!      DO l = 1, nz
!        WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES12.4, ')' )" ) &
!         ROW( l ), COL( l ), VAL_est( l )
!      END DO
       WRITE( 6, "( ' Successful run with ', I0, ' differences, error = ',     &
      &   ES7.1 )" ) m, MAXVAL( ABS( VAL_est - VAL ) )
     END IF
     CALL SHA_estimate( n, nz, ROW, COL, m, S, n, m, Y, n, m, VAL_est,         &
                        data, control, inform, ORDER = ORDER )
     IF ( inform%status /= 0 ) THEN ! Failure
       WRITE( 6, "( ' return with nonzero status ',I0,' from SHA_estimate' )" )&
         inform%status
     ELSE
!      WRITE( 6, "( /, ' Successful run with ', I0,                            &
!     &             ' differences, estimated matrix:' )" ) m
!      DO l = 1, nz
!        WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES12.4, ')' )" ) &
!         ROW( l ), COL( l ), VAL_est( l )
!      END DO
       WRITE( 6, "( ' Successful run with ', I0, ' differences, error = ',     &
      &   ES7.1 )" ) m, MAXVAL( ABS( VAL_est - VAL ) )
     END IF
     WRITE( 6, "( ' Test with insufficient data ...' )" )
     control%extra_differences = 1
     CALL SHA_estimate( n, nz, ROW, COL, 1_ip_, S, n, m, Y, n, m, VAL_est,     &
                        data, control, inform )
     IF ( inform%status < 0 ) THEN ! Failure
       WRITE( 6, "( ' return with nonzero status ',I0,' from SHA_estimate' )" )&
         inform%status
     ELSE IF ( inform%status > 0 ) THEN ! Warning
       WRITE( 6, "( ' Warning = ', I0, ' run with ', I0, ' differences,',      &
      & ' error = ', ES7.1 )" ) inform%status, 1, MAXVAL( ABS( VAL_est - VAL ) )
     ELSE
!      WRITE( 6, "( /, ' Successful run with ', I0,                            &
!     &             ' differences, estimated matrix:' )" ) m
!      DO l = 1, nz
!        WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES12.4, ')' )" ) &
!         ROW( l ), COL( l ), VAL_est( l )
!      END DO
       WRITE( 6, "( ' Successful run with ', I0, ' differences, error = ',     &
      &   ES7.1 )" ) 1, MAXVAL( ABS( VAL_est - VAL ) )
     END IF
     CALL SHA_terminate( data, control, inform ) ! Delete internal workspace
     DEALLOCATE( S, Y, ORDER )
   END DO
   WRITE( 6, "( /, ' TODO: test program is not yet exhausive' )" )

   END PROGRAM GALAHAD_SHA_test_deck
