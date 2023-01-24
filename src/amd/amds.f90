! THIS VERSION: GALAHAD 4.1 - 2022-10-11 AT 11:30 GMT.
   PROGRAM GALAHAD_AMD_EXAMPLE
   USE GALAHAD_AMD_double
   IMPLICIT NONE
   TYPE ( AMD_data_type ) :: data
   TYPE ( AMD_control_type ) :: control
   TYPE ( AMD_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: nz = 10
   INTEGER, DIMENSION( n + 1 ) :: PTR = (/ 1, 3, 6, 9, 10, 11 /)
   INTEGER, DIMENSION( nz ) :: ROW = (/ 1, 2, 2, 3, 5, 3, 4, 5, 4, 5 /)
   INTEGER, DIMENSION( n ) :: PERM
   CALL AMD_initialize( data, control, inform ) ! Initialize control parameters
   CALL AMD_order( n, PTR, ROW, PERM, data, control, inform ) ! find permutation
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' AMD: permutation =', /, 10( I5 ) )" ) PERM
   ELSE                                         !  Error returns
     WRITE( 6, "( ' AMD_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL AMD_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_AMD_EXAMPLE
