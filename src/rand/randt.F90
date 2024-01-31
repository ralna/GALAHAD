! THIS VERSION: GALAHAD 4.3 - 2024-01-31 AT 08:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_RAND_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_RAND_precision
   IMPLICIT NONE
   TYPE (RAND_seed) seed
   INTEGER ( KIND = ip_ ) :: random_integer, value
   REAL ( kind = rp_ ) :: random_real
!  Initialize the generator word
   CALL RAND_initialize( seed )  !  Get the current generator word
   CALL RAND_get_seed( seed, value )
   WRITE( 6, "( ' generator word = ', I0 )" ) value
!  Generate a random real in [-1, 1]
   CALL RAND_random_real( seed, .FALSE., random_real )
   WRITE( 6, "( ' random real = ', F5.2 )" ) random_real
!  Generate another random real
   CALL RAND_random_real( seed, .FALSE., random_real )
   WRITE( 6, "( ' second random real = ', F5.2 )" ) random_real
!  Restore the generator word
   CALL RAND_set_seed( seed, value )
!  Generate a random integer in [1, 100]
   CALL RAND_random_integer( seed, 100_ip_, random_integer )
   WRITE( 6, "( ' random integer in [1,100] = ', I0 )" ) random_integer
!  Generate another random integer
   CALL RAND_random_integer( seed, 100_ip_, random_integer )
   WRITE( 6, "( ' second random integer in [1,100] = ', I0 )" ) random_integer
   END PROGRAM GALAHAD_RAND_test
