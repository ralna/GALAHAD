! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-  G A L A H A D  -  D U M M Y   M U M P S   S U B R O U T I N E  -*-*-*-

      SUBROUTINE MUMPS_precision( mumps_par )
      USE GALAHAD_KINDS_precision
      USE GALAHAD_MUMPS_TYPES_precision
      IMPLICIT NONE
      TYPE ( MUMPS_STRUC ) :: mumps_par
      mumps_par%INFOG( 1 ) = - 999  ! error code
      RETURN
      END SUBROUTINE MUMPS_precision
