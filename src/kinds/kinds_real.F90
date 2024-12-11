! THIS VERSION: GALAHAD 5.1 - 2024-12-11 AT 14:40 GMT

#include "galahad_modules.h"

!-*-*-*-*-  G A L A H A D _  K I N D S _ R E A L   M O D U L E   -*-*-*-*-*-

!  Copyright reserved, GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released, GALAHAD Version 5.1, December 11th 2024

MODULE GALAHAD_KINDS_precision
  USE GALAHAD_KINDS
  IMPLICIT NONE
  PUBLIC

!---------------------
!  R e a l  k i n d s
!---------------------

#ifdef REAL_32
  INTEGER, PARAMETER :: real_bytes_ = 4
  INTEGER, PARAMETER :: rp_ = r4_
  INTEGER, PARAMETER :: cp_ = c4_
  INTEGER, PARAMETER :: rpc_ = spc_
#elif REAL_128
  INTEGER, PARAMETER :: real_bytes_ = 16
  INTEGER, PARAMETER :: rp_ = r16_
  INTEGER, PARAMETER :: cp_ = c16_
  INTEGER, PARAMETER :: rpc_ = qpc_
#else
  INTEGER, PARAMETER :: real_bytes_ = 8
  INTEGER, PARAMETER :: rp_ = r8_
  INTEGER, PARAMETER :: cp_ = c8_
  INTEGER, PARAMETER :: rpc_ = dpc_
#endif

END MODULE GALAHAD_KINDS_precision
