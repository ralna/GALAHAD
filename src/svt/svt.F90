! THIS VERSION: GALAHAD 5.1 - 2024-11-12 AT 14:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*  G A L A H A D _ S V T   M O D U L E  *-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 4.2 October 25th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SVT_precision

     USE GALAHAD_KINDS_precision

!  ==========================
!  Sparse vector derived type
!  ==========================

     TYPE, PUBLIC :: SVT_type
       INTEGER ( KIND = ip_ ) :: ne
       LOGICAL :: sparse = .TRUE.
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ind
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: val
     END TYPE SVT_type

!  End of module GALAHAD_SVT

   END MODULE GALAHAD_SVT_precision
