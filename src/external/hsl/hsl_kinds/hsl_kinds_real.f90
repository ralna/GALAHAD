! THIS VERSION: HSL SUBSET 1.1 - 2024-12-11 AT 14:15 GMT

#include "hsl_subset.h"

!-*-*-*-  H S L _ S U B S E T _  K I N D S _ R E A L   M O D U L E   -*-*-*-

MODULE HSL_KINDS_real
  USE HSL_KINDS
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

END MODULE HSL_KINDS_real
