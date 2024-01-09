! THIS VERSION: GALAHAD 4.2 - 2023-11-10 AT 09:00 GMT.

#include "galahad_modules.h"

   PROGRAM SVT_test_deck
   USE GALAHAD_SVT_precision
   TYPE ( SVT_type ) :: V
   INTEGER ( KIND = ip_ ) :: status
   V%ne = 1

   ALLOCATE( V%ind( V%ne ), V%val( V%ne ), stat = status )
   V%ind( 1 ) = 1 ; V%val( 1 ) = 1.0

   WRITE( 6, "( A, I3, /, A, I3, ES10.2 )" ) ' ne = ', V%ne,                   &
    ' ind, val = ', V%ind( 1 ), V%val( 1 )

   DEALLOCATE( V%ind, V%val )
   END PROGRAM SVT_test_deck
