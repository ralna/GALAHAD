! THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 09:00 GMT.

#include "galahad_modules.h"

     PROGRAM NODEND_test
     USE GALAHAD_KINDS, ONLY: ip_
     USE GALAHAD_NODEND_precision
     USE GALAHAD_SMT_precision
     INTEGER, PARAMETER :: out = 6
!    INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, nz = 5, nz_compact = 4
!    INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: PTR = (/ 1, 3, 4, 6 /)
!    INTEGER ( KIND = ip_ ), DIMENSION( nz ) :: IND = (/ 1, 3, 2, 1, 3 /)
     INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, nz = 2, nz_compact = 4
     INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: PTR = (/ 1, 2, 2, 3 /)
     INTEGER ( KIND = ip_ ), DIMENSION( nz ) :: IND = (/ 3, 1 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n ) :: PERM
     TYPE ( SMT_type ) :: A
     TYPE ( NODEND_control_type ) :: control
     TYPE ( NODEND_inform_type ) :: inform
     INTEGER ( KIND = ip_ ) :: version, type, status

!  test the full storage versions

     DO version = 1, 3
!      IF ( version == 2 ) CYCLE
       SELECT CASE( version )
       CASE( 1 )
         control%version = '4.0'
       CASE( 2 )
         control%version = '5.1'
       CASE( 3 )
         control%version = '5.2'
       END SELECT
       CALL NODEND_order_adjacency( n, PTR, IND, PERM, control, inform )
       IF ( PERM( 1 ) <= 0 ) THEN
         WRITE( out, "( ' No Nodend ', A, ' available, stopping' )" )          &
           TRIM( control%version )
       ELSE IF ( inform%status < 0 ) THEN
         WRITE( out, "( ' Nodend ', A, ' failure, status = ', I0 )" )          &
           TRIM( control%version ), inform%status
       ELSE 
         IF ( inform%status == 0 ) THEN
           WRITE( out, "(  ' Nodend ', A, ' call successful' )" )              &
             TRIM( control%version )
         ELSE
           WRITE( out, "(  ' Nodend ', A, ' call unsuccessful,',               &
          &  ' no permutation found' )" ) TRIM( control%version )
         END IF
       END IF
     END DO

!  test the half storage versions

     ALLOCATE( A%row( nz_compact ), A%col( nz_compact ), A%ptr( n + 1 ),       &
               STAT = status )
     A%n = n ; A%ne = nz_compact
     A%row = (/ 1, 2, 3, 3 /)
     A%col = (/ 1, 2, 1, 3 /)
     A%ptr = (/ 1, 2, 3, 5 /)

     DO version = 1, 3
       SELECT CASE( version )
       CASE( 1 )
         control%version = '4.0'
       CASE( 2 )
         control%version = '5.1'
       CASE( 3 )
         control%version = '5.2'
       END SELECT
       CALL NODEND_half_order( n, A%ptr, A%col, PERM, control, inform )
       IF ( PERM( 1 ) <= 0 ) THEN
         WRITE( out, "( ' No Nodend_half ', A, ' available, stopping' )" )     &
           TRIM( control%version )
       ELSE IF ( inform%status < 0 ) THEN
         WRITE( out, "( ' Nodend_half ', A, ' failure, status = ', I0 )" )     &
           TRIM( control%version ), inform%status
       ELSE 
         IF ( inform%status == 0 ) THEN
           WRITE( out, "(  ' Nodend_half ', A, ' call successful' )" )         &
             TRIM( control%version )
         ELSE
           WRITE( out, "(  ' Nodend_half ', A, ' call unsuccessful,',          &
          &  ' no permutation found' )" ) TRIM( control%version )
         END IF
       END IF
     END DO

!  test the different storage versions

     DO type = 1, 3
       SELECT CASE( type )
       CASE( 1 )
         CALL SMT_put( A%type, 'COORDINATE', status )
       CASE( 2 )
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', status )
       CASE( 3 )
         CALL SMT_put( A%type, 'DENSE', status )
       END SELECT
       CALL NODEND_order( A, PERM, control, inform )
       IF ( PERM( 1 ) <= 0 ) THEN
         WRITE( out, "( ' No Nodend ', A, ' available, stopping' )" )          &
           TRIM( control%version )
       ELSE IF ( inform%status < 0 ) THEN
         WRITE( out, "( ' Nodend ', A, ' storage failure, status = ', I0 )" )  &
           TRIM( SMT_get( A%type ) ), inform%status
       ELSE 
         IF ( inform%status == 0 ) THEN
           WRITE( out, "(  ' Nodend ', A, ' storage call successful' )" )      &
             TRIM( SMT_get( A%type ) )
         ELSE
           WRITE( out, "(  ' Nodend ', A, ' storage call unsuccessful,',       &
          &  ' no permutation found' )" ) TRIM( SMT_get( A%type ) )
         END IF
       END IF
       DEALLOCATE( A%type )
     END DO

     DEALLOCATE( A%row, A%col, A%ptr )

     END PROGRAM NODEND_test
