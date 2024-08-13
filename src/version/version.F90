! THIS VERSION: GALAHAD 5.1 - 2024-08-13 AT 13:15 GMT.

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

      CHARACTER ( LEN = 3 ) :: major = '5  '
      CHARACTER ( LEN = 3 ) :: minor = '0  '
      CHARACTER ( LEN = 5 ) :: patch = '0    '

    CONTAINS

!-*-*-*-*-*-*-  G A L A H A D   V E R S I O N   F U N C T I O N   -*-*-*-*-*-*-

      FUNCTION VERSION(  )
      CHARACTER ( LEN = 13 ) :: version

!  return the current GALAHAD version number (major.minor.patch)

      version = REPEAT( ' ', 13 )
      version = TRIM( major ) // '.' // TRIM( minor ) // '.' // TRIM( patch ) 

!  End of subroutine VERSION

      END FUNCTION VERSION

!  End of module GALAHAD_VERSION

    END MODULE GALAHAD_VERSION

