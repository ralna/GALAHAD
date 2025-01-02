! THIS VERSION: GALAHAD 5.1 - 2024-08-16 AT 10:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-  G A L A H A D _  V E R S I O N    C   I N T E R F A C E  -*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 5.1. August 16th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_VERSION_ciface
    USE GALAHAD_KINDS
    USE GALAHAD_VERSION, ONLY: f_VERSION_galahad => VERSION_galahad

  CONTAINS

!  --------------------------------------
!  C interface to fortran version_galahad
!  --------------------------------------

    SUBROUTINE version_galahad( major_version, minor_version, patch_version )  &
                                BIND( C )
    IMPLICIT NONE

!  dummy arguments

    INTEGER ( KIND = ipc_ ) :: major_version, minor_version, patch_version
    CALL f_VERSION_galahad( major_version, minor_version, patch_version )

    RETURN

    END SUBROUTINE version_galahad

  END MODULE GALAHAD_VERSION_ciface
