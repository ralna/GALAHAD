! THIS VERSION: GALAHAD 5.1 - 2024-12-12 AT 11:20 GMT

#ifdef INTEGER_64
#define SPRAL_KINDS SPRAL_KINDS_64
#ifdef REAL_32
#define SPRAL_KINDS_precision SPRAL_KINDS_single_64
#elif REAL_128
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple_64
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_double_64
#endif
#else
#ifdef REAL_32
#define SPRAL_KINDS_precision SPRAL_KINDS_single
#elif REAL_128
#define SPRAL_KINDS_precision SPRAL_KINDS_quadruple
#else
#define SPRAL_KINDS_precision SPRAL_KINDS_double
#endif
#endif

!-*-*-*-*-*-*-*-  S P R A L _ K I N D S _ R E A L  M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   separated from SPRAL_KINDS, GALAHAD Version 5.1, December 12th 2024

MODULE SPRAL_KINDS_precision
  USE SPRAL_KINDS
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
  INTEGER, PARAMETER :: C_RP_ = rpc_
#elif REAL_128
  INTEGER, PARAMETER :: real_bytes_ = 16
  INTEGER, PARAMETER :: rp_ = r16_
  INTEGER, PARAMETER :: cp_ = c16_
  INTEGER, PARAMETER :: rpc_ = qpc_
  INTEGER, PARAMETER :: C_RP_ = rpc_
#else
  INTEGER, PARAMETER :: real_bytes_ = 8
  INTEGER, PARAMETER :: rp_ = r8_
  INTEGER, PARAMETER :: cp_ = c8_
  INTEGER, PARAMETER :: rpc_ = dpc_
  INTEGER, PARAMETER :: C_RP_ = rpc_
#endif

END MODULE SPRAL_KINDS_precision
