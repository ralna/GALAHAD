! THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*- G A L A H A D _ M U M P S _ T Y P E S   M O D U L E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. November 2nd 2022

   MODULE GALAHAD_MUMPS_TYPES_precision

#ifdef REAL_128
     USE GALAHAD_KINDS, ONLY : ip_, long_, r16_
#else
     USE GALAHAD_KINDS, ONLY : ip_, long_
#endif
     IMPLICIT NONE
     PUBLIC
     INTEGER ( KIND = ip_ ), PARAMETER :: MPI_COMM_WORLD = 0

!  include the current mumps derived types

#ifdef REAL_32
     INCLUDE 'smumps_struc.h'
#elif REAL_128
     INCLUDE 'qmumps_struc.h'
#else
     INCLUDE 'dmumps_struc.h'
#endif

!  End of module GALAHAD_MUMPS_TYPES_double

   END MODULE GALAHAD_MUMPS_TYPES_precision
