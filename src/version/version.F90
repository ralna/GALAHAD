! THIS VERSION: GALAHAD 5.1 - 2024-08-14 AT 09:45 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D    V E R S I O N   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban/Toint, 
!  for GALAHAD productions
!  Principal author: Nick Gould
!
!  History -
!   originally released GALAHAD Version 5.1. August 13th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_VERSION

      USE GALAHAD_KINDS_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: VERSION

      INTEGER, PARAMETER :: major = 5
      INTEGER, PARAMETER :: minor = 0
      INTEGER, PARAMETER :: patch = 0

    CONTAINS

!-*-*-*-*-*-*-  G A L A H A D   V E R S I O N   F U N C T I O N   -*-*-*-*-*-*-

      FUNCTION VERSION(  )
      INTEGER, DIMENSION( 3 ) :: version

!  return the current GALAHAD version number (major.minor.patch)

      version = (/ major, minor, patch /)

!  End of subroutine VERSION

      END FUNCTION VERSION

!  End of module GALAHAD_VERSION

    END MODULE GALAHAD_VERSION

