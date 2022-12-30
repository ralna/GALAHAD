! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*- G A L A H A D _ M U M P S _ T Y P E S   M O D U L E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. November 2nd 2022

   MODULE GALAHAD_MUMPS_TYPES_precision

     USE GALAHAD_KINDS, ONLY : ip_
     IMPLICIT NONE
     PUBLIC
     INTEGER ( KIND = ip_ ), PARAMETER :: MPI_COMM_WORLD = 0

!  include the current mumps derived types

#ifdef GALAHAD_SINGLE
     INCLUDE 'smumps_struc.h'
#else
     INCLUDE 'dmumps_struc.h'
#endif

!  End of module GALAHAD_MUMPS_TYPES_double

   END MODULE GALAHAD_MUMPS_TYPES_precision
