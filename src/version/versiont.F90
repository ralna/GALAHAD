! THIS VERSION: GALAHAD 5.1  - 2024-08-14 AT 14:45 GMT.

#include "galahad_modules.h"

   PROGRAM GALAHAD_VERSION_TEST
   USE GALAHAD_KINDS
   USE GALAHAD_VERSION
   IMPLICIT NONE

   INTEGER ( KIND = ip_ ) :: major, minor, patch
   CALL VERSION_galahad( major, minor, patch )

   WRITE( 6, "(' current GALAHAD version is ', I0, '.',  I0, '.',  I0 )" )     &
     major, minor, patch

   END PROGRAM GALAHAD_VERSION_TEST
