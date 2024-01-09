! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D _ U S E R D A T A   M O D U L E -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as part of NLPT GALAHAD Version 2.0. February 16th 2005
!   seprated into its own module May 3rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_USERDATA_precision

     USE GALAHAD_KINDS_precision

     IMPLICIT NONE

     PUBLIC

!  ======================================
!  The GALAHAD_userdata_type derived type
!  ======================================

     TYPE, PUBLIC :: GALAHAD_userdata_type
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: integer
       REAL( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: real
       COMPLEX ( KIND = cp_ ), ALLOCATABLE, DIMENSION( : ) :: complex
       CHARACTER, ALLOCATABLE, DIMENSION( : ) :: character
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical
       INTEGER ( KIND = ip_ ), POINTER,                                        &
         DIMENSION( : ) :: integer_pointer => NULL( )
       REAL( KIND = rp_ ), POINTER, DIMENSION( : ) :: real_pointer => NULL( )
       COMPLEX ( KIND = cp_ ), POINTER,                                        &
         DIMENSION( : ) :: complex_pointer => NULL( )
       CHARACTER, POINTER, DIMENSION( : ) :: character_pointer => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: logical_pointer => NULL( )
     END TYPE GALAHAD_userdata_type

!  End of module GALAHAD_USERDATA

   END MODULE GALAHAD_USERDATA_precision
