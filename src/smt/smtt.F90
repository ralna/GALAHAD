! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

   PROGRAM SMT_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SMT_precision
   TYPE ( SMT_type ) :: A
   INTEGER ( KIND = ip_ ) :: s
   A%m = 1 ; A%n = 1 ; A%ne = 1

   ALLOCATE( A%row( A%m ), A%col( A%n ), A%val( A%ne ), A%ptr( 1 ) )
   CALL SMT_put( A%id, 'mat', s )
   CALL SMT_put( A%type, 'symm', s )
   A%row( 1 ) = 1 ; A%col( 1 ) = 1 ; A%ptr( 1 ) = 1 ; A%val( 1 ) = 1.0

   WRITE( 6, "( A, 3I3, /, 2A, 1X, A /, A, 3I3, ES10.2 )" )                   &
    ' m, n, ne = ', A%m, A%n, A%ne,                                           &
    ' id, type = ', SMT_get( A%id ), SMT_get( A%type ),                       &
    ' row, col, ptr, val = ', A%row( 1 ), A%col( 1 ), A%ptr( 1 ), A%val( 1 )

   DEALLOCATE( A%row, A%col, A%val, A%ptr, A%id, A%type )
   END PROGRAM SMT_test_deck
