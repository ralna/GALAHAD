   PROGRAM ICFS_EXAMPLE   !  GALAHAD 4.1 - 2022-12-04 AT 09:30 GMT.
   USE GALAHAD_ICFS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( ICFS_data_type ) :: data
   TYPE ( ICFS_control_type ) control
   TYPE ( ICFS_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: ne = 5
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: PTR, ROW
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )  :: DIAG, VAL
   REAL ( KIND = wp ) :: X( n )
! allocate and set lower triangle of matrix in sparse by column form
   ALLOCATE( PTR( n + 1 ), DIAG( n ), VAL( ne ), ROW( ne ) )
   PTR = (/ 1, 2, 4, 5, 5, 5 /)
   ROW = (/ 2, 3, 5, 4 /)
   DIAG = (/ 2.0_wp, 5.0_wp, 1.0_wp, 7.0_wp, 2.0_wp /)
   VAL = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
! problem setup complete
   CALL ICFS_initialize( data, control, inform )
   control%icfs_vectors = 1
! form and factorize the preconditioner, P = L L^T
   CALL ICFS_factorize( n, PTR, ROW, DIAG, VAL, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of ICFS_factorize with status = ', inform%status
     STOP
   END IF
! use the factors to solve L L^T x = b, with b input in x
   X( : n ) = (/ 3.0_wp, 8.0_wp, 3.0_wp, 8.0_wp, 3.0_wp /)
   CALL ICFS_triangular_solve( n, X, .FALSE., data, control, inform )
   CALL ICFS_triangular_solve( n, X, .TRUE., data, control, inform )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( ' ICFS - Preconditioned solution is ', 5F6.2 )" ) X
   ELSE
     WRITE( 6, "( ' ICFS - exit status = ', I0 )" ) inform%status
   END IF
! clean up
   CALL ICFS_terminate( data, control, inform )
   DEALLOCATE( DIAG, VAL, ROW, PTR )
   STOP
   END PROGRAM ICFS_EXAMPLE
