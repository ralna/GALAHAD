! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
      PROGRAM GALAHAD_RAND_test_deck

!  Comprehensive test deck for module RAND

      USE GALAHAD_RAND_double
      IMPLICIT NONE

      INTEGER :: random_integer, seed
      REAL ( kind = KIND( 1.0D+0 ) ) :: random_real
      CHARACTER ( LEN = * ), PARAMETER :: C_R = "( ' r = ', F10.2 )"  
      CHARACTER ( LEN = * ), PARAMETER :: C_I = "( ' i = ', I10 )"

      CALL RAND_RANDOM_REAL( .TRUE., random_real )
      WRITE( 6, C_R ) random_real
      CALL RAND_RANDOM_REAL( .FALSE., random_real )
      WRITE( 6, C_R ) random_real
      CALL RAND_RANDOM_INTEGER( 10, random_integer )
      WRITE( 6, C_I ) random_integer
      CALL RAND_RANDOM_INTEGER( - 10, random_integer )
      WRITE( 6, C_I ) random_integer
      CALL RAND_GET_SEED( seed )
      WRITE( 6, C_I ) seed
      CALL RAND_SET_SEED( 10 )
      CALL RAND_GET_SEED( seed )
      WRITE( 6, C_I ) seed

      STOP 
      END PROGRAM GALAHAD_RAND_test_deck
