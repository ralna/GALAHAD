! THIS VERSION: GALAHAD 5.1 - 2024-12-10 AT 10:35 GMT

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

!-*-*-*-*-*-*-*-*-*-  S P R A L _ K I N D S   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released, GALAHAD Version 4.1, December 6th 2022

MODULE SPRAL_KINDS
  USE ISO_C_BINDING
  USE ISO_FORTRAN_ENV
  IMPLICIT NONE
  PUBLIC

!---------------------------------------------
!   R e a l  a n d   i n t e g e r   t y p e s
!---------------------------------------------

!  basic kinds used

  INTEGER, PARAMETER :: i4_ = INT32
  INTEGER, PARAMETER :: i8_ = INT64
  INTEGER, PARAMETER :: r4_ = REAL32
  INTEGER, PARAMETER :: r8_ = REAL64
  INTEGER, PARAMETER :: c4_ = KIND( ( 1.0_r4_, 1.0_r4_ ) )
  INTEGER, PARAMETER :: c8_ = KIND( ( 1.0_r8_, 1.0_r8_ ) )

!  if 128 bit reals are supported, use them

#ifdef REAL_128
  INTEGER, PARAMETER :: r16_ = REAL128
  INTEGER, PARAMETER :: c16_ = KIND( ( 1.0_r16_, 1.0_r16_ ) )
  INTEGER, PARAMETER :: qpc_ = C_FLOAT128
#endif

!  common aliases

  INTEGER, PARAMETER :: sp_ = r4_
  INTEGER, PARAMETER :: dp_ = r8_
  INTEGER, PARAMETER :: long_ = i8_
  INTEGER, PARAMETER :: spc_ = C_FLOAT
  INTEGER, PARAMETER :: dpc_ = C_DOUBLE
  INTEGER, PARAMETER :: longc_ = C_INT64_T
  INTEGER, PARAMETER :: lp_ = KIND( .TRUE. )

!  integer kinds

#ifdef INTEGER_64
  INTEGER, PARAMETER :: ip_ = INT64
  INTEGER, PARAMETER :: ipc_ = C_INT64_T
#else
  INTEGER, PARAMETER :: ip_ = INT32
  INTEGER, PARAMETER :: ipc_ = C_INT32_T
#endif

  INTEGER, PARAMETER :: C_IP_ = ipc_
  INTEGER, PARAMETER :: CLONG_ = longc_

END MODULE SPRAL_KINDS

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
