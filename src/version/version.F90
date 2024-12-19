! THIS VERSION: GALAHAD 5.1 - 2024-12-12 AT 09:25 GMT.

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
      USE GALAHAD_KINDS
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: VERSION_galahad

      INTEGER ( KIND = ip_ ), PARAMETER :: major = 5
      INTEGER ( KIND = ip_ ), PARAMETER :: minor = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: patch = 0

    CONTAINS

!-*-  G A L A H A D   V E R S I O N _ G A L A H A D   S U B R O U T I N E   -*-

      SUBROUTINE VERSION_galahad( major_version, minor_version, patch_version ) 
      INTEGER ( KIND = ip_ ) :: major_version, minor_version, patch_version

!  return the current GALAHAD version number (major.minor.patch)

      major_version = major
      minor_version = minor
      patch_version = patch
      RETURN

!  End of subroutine VERSION_galahad

      END SUBROUTINE VERSION_galahad

!  End of module GALAHAD_VERSION

    END MODULE GALAHAD_VERSION
