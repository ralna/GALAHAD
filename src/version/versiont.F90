! THIS VERSION: GALAHAD 5.1  - 2024-08-14 AT 09:45 GMT.

#include "galahad_modules.h"

   PROGRAM GALAHAD_VERSION_TEST
   USE GALAHAD_VERSION
   IMPLICIT NONE
   INTEGER, DIMENSION( 3 ) :: current
   current = VERSION( )
   WRITE( 6, "(' current GALAHAD version is ', I0, '.',  I0, '.',  I0 )" )     &
     current
   END PROGRAM GALAHAD_VERSION_TEST
