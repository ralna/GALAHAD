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

      SUBROUTINE VERSION(major_version, minor_version, patch_version) BIND(C, NAME="version_galahad")
      INTEGER :: major_version, minor_version, patch_version

!  return the current GALAHAD version number (major.minor.patch)
      major_version = major
      minor_version = minor
      patch_version = patch
      RETURN

!  End of subroutine VERSION

      END SUBROUTINE VERSION

!  End of module GALAHAD_VERSION

    END MODULE GALAHAD_VERSION
