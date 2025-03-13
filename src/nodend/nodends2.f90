! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 09:00 GMT.
     PROGRAM NODEND_example_adjacency
     USE GALAHAD_KINDS_double, ONLY: ip_
     USE GALAHAD_NODEND_double
     INTEGER, PARAMETER :: out = 6
     INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, nz = 6
     INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: PTR = (/ 1, 3, 3, 4, 5, 7 /)
     INTEGER ( KIND = ip_ ), DIMENSION( nz ) :: IND = (/ 3, 5, 1, 5, 1, 4 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM
     TYPE ( NODEND_control_type ) :: control
     TYPE ( NODEND_inform_type ) :: inform
     control%version = '5.1'
     CALL NODEND_order_adjacency( n, PTR, IND, PERM, control, inform )
     IF ( PERM( 1 ) <= 0 ) THEN
       WRITE( out, "( ' No METIS ', A, ' available, stopping' )" )            &
         control%version
     ELSE IF ( inform%status < 0 ) THEN
       WRITE( out, "( ' Nodend ', A, ' failure, status = ', I0 )" )            &
         control%version, inform%status
     ELSE 
       IF ( inform%status == 0 ) THEN
         WRITE( out, "( ' Nodend ', A, ' order_adjacency call successful' )" ) &
           TRIM( control%version )
         WRITE( out, "( ' permutation =', 5I2 )" ) PERM
       ELSE
         WRITE( out, "(  ' Nodend ', A, ' order call unsuccessful,',           &
        &  ' no permutation found' )" ) TRIM( control%version )
       END IF
     END IF
     END PROGRAM NODEND_example_adjacency
