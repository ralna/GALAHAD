! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 09:00 GMT.
     PROGRAM NODEND_example
     USE GALAHAD_KINDS_double, ONLY: ip_
!    USE GALAHAD_SMT_double
     USE GALAHAD_NODEND_double
     INTEGER, PARAMETER :: out = 6
     INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, ne = 8
     INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM
     TYPE ( SMT_type ) :: A
     TYPE ( NODEND_control_type ) :: control
     TYPE ( NODEND_inform_type ) :: inform
     INTEGER :: smt_stat
     CALL SMT_put( A%type, 'COORDINATE', smt_stat )
     A%n = n ; A%ne = ne
     ALLOCATE( A%row( ne ), A%col( ne ) )
     A%row = (/ 1, 2, 3, 3, 4, 5, 5, 5 /)
     A%col = (/ 1, 2, 1, 3, 4, 1, 4, 5 /)
     control%version = '5.1'
     CALL NODEND_order( A, PERM, control, inform )
     IF ( PERM( 1 ) <= 0 ) THEN
       WRITE( out, "( ' No METIS ', A, ' available, stopping' )" )            &
         control%version
     ELSE IF ( inform%status < 0 ) THEN
       WRITE( out, "( ' Nodend ', A, ' failure, status = ', I0 )" )            &
         control%version, inform%status
     ELSE 
       IF ( inform%status == 0 ) THEN
         WRITE( out, "( ' Nodend ', A, ' order call successful' )" )           &
           TRIM( control%version )
         WRITE( out, "( ' permutation =', 5I2 )" ) PERM
       ELSE
         WRITE( out, "(  ' Nodend ', A, ' order call unsuccessful,',           &
        &  ' no permutation found' )" ) TRIM( control%version )
       END IF
     END IF
     DEALLOCATE( A%row, A%col, A%type )
     END PROGRAM NODEND_example
