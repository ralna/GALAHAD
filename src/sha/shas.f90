! THIS VERSION: GALAHAD 4.1 - 2023-08-19 AT 15:40 GMT.
   PROGRAM GALAHAD_SHA_EXAMPLE
   USE GALAHAD_SHA_double  ! double precision version
   USE GALAHAD_RAND_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SHA_data_type ) :: data
   TYPE ( SHA_control_type ) :: control
   TYPE ( SHA_inform_type ) :: inform
   INTEGER :: i, j, k, k_s, l
   REAL ( KIND = wp ) ::  v
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: ORDER
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: S, Y
   TYPE ( RAND_seed ) :: seed
   INTEGER, PARAMETER :: n = 5, nz = 9  ! set problem data
   INTEGER :: ROW( nz ), COL( nz )
   REAL ( KIND = wp ) ::  VAL( nz ), VAL_est( nz )
   ROW = (/ 1, 1, 1, 1, 1, 2, 3, 4, 5 /)  ! N.B. upper triangle only
   COL = (/ 1, 2, 3, 4, 5, 2, 3, 4, 5 /)
   VAL = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 /) ! artificial values
   CALL SHA_initialize( data, control, inform )  ! initialize
   control%approximation_algorithm = 2 ! symmetric approximation
   CALL SHA_analyse( n, nz, ROW, COL, data, control, inform ) ! analyse sparsity
   IF ( inform%status /= 0 ) THEN             ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_analyse' )" ) &
       inform%status ; STOP
   END IF
   WRITE( 6, "( 1X, I0, ' differences are needed,',                            &
  & ' one or more extra might help' )" ) inform%differences_needed
   control%extra_differences = 1 ! use as many differences as required + 1
   k_s = inform%differences_needed + control%extra_differences
!  artifical setup: compute random s_i and then form y_i = Hessian * s_i
   ALLOCATE( S( n, k_s ), Y( n, k_s ), ORDER( k_s ) )
   CALL RAND_initialize( seed )
   DO k = 1, k_s
     DO i = 1, n  ! choose random S
       CALL RAND_random_real( seed, .FALSE., S( i, k ) )
       CALL RAND_random_real( seed, .FALSE., Y( i, k ) )
       Y( i, k ) = Y( i, k ) * 0.001
     END DO
     Y( : n, k ) = 0.0_wp  ! form Y = H * S
     DO l = 1, nz
       i = ROW( l ) ; j = COL( l ) ; v = VAL( l )
       Y( i, k ) = Y( i, k ) + v * S( j, k )
       IF ( i /= j ) Y( j, k ) = Y( j, k ) + v * S( i, k )
     END DO
     ORDER( k ) = k_s - k + 1  ! pick the (s,y) vectors in reverse order
   END DO
!  approximate the Hessian
   CALL SHA_estimate( n, nz, ROW, COL, k_s, S, n, k_s, Y, n, k_s, VAL_est,     &
                      data, control, inform, ORDER = ORDER )
   IF ( inform%status /= 0 ) THEN  ! Failure
     WRITE( 6, "( ' return with nonzero status ', I0, ' from SHA_estimate' )" )&
       inform%status ; STOP
   ELSE
     WRITE( 6, "( /, ' Successful run with ', I0,                              &
    &             ' differences, estimated matrix:' )" ) k_s
     DO l = 1, nz
       WRITE( 6, "( ' (row,col,val) = (', I0, ',', I0, ',', ES9.2, ')' )" )    &
        ROW( l ), COL( l ), VAL_est( l )
     END DO
   END IF
   CALL SHA_terminate( data, control, inform ) ! Delete internal workspace
   END PROGRAM GALAHAD_SHA_EXAMPLE

