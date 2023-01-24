! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_AMD_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_AMD_precision
   IMPLICIT NONE
   TYPE ( AMD_data_type ) :: data
   TYPE ( AMD_control_type ) :: control
   TYPE ( AMD_inform_type ) :: inform
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: nz = 10
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: PTR = (/ 1, 3, 6, 9, 10, 11 /)
   INTEGER ( KIND = ip_ ), DIMENSION( nz ) :: ROW = (/ 2, 1, 5, 3, 2, 5, 4, &
                                                       3, 4, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM
   CALL AMD_initialize( data, control, inform ) ! Initialize control parameters
   CALL AMD_order( n, PTR, ROW, PERM, data, control, inform ) ! find permutation
   WRITE( 6, "( ' Aggressive absorption' )" )
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' AMD: permutation =', /, 10( I5 ) )" ) PERM
   ELSE                                         !  Error returns
     WRITE( 6, "( ' AMD_solve exit status = ', I6 ) " ) inform%status
   END IF
   control%aggressive = .FALSE.
   WRITE( 6, "( ' Non-aggressive absorption' )" )
   CALL AMD_order( n, PTR, ROW, PERM, data, control, inform ) ! find permutation
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' AMD: permutation =', /, 10( I5 ) )" ) PERM
   ELSE                                         !  Error returns
     WRITE( 6, "( ' AMD_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL AMD_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_AMD_EXAMPLE
