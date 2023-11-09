! THIS VERSION: GALAHAD 4.2 - 2023-10-25 AT 16:30 GMT.

#include "galahad_modules.h"

   PROGRAM SVT_test_deck
   USE GALAHAD_SVT_precision
   TYPE ( SVT_type ) :: A
   A%n = 1 ; A%ne = 1

   ALLOCATE( A%ind( A%ne ), A%val( A%ne ) )
   A%ind( 1 ) = 1 ; A%val( 1 ) = 1.0

   WRITE( 6, "( A, 2I3, /, A, 3I3, ES10.2 )" ) ' n, ne = ', A%n, A%ne,         &
    ' ind, col, ptr, val = ', A%ind( 1 ), A%val( 1 )

   DEALLOCATE( A%ind, A%val )
   END PROGRAM SVT_test_deck
