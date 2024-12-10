! THIS VERSION: GALAHAD 5.1 - 2024-12-10 AT 10:30 GMT

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ K I N D S   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released, GALAHAD Version 4.1, December 6th 2022

MODULE GALAHAD_KINDS
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
  INTEGER, PARAMETER :: qp_ = r16_
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

!  integer and logical kinds (replace the latter in fortran 2023)

#ifdef INTEGER_64
  INTEGER, PARAMETER :: ip_ = INT64
  INTEGER, PARAMETER :: ipc_ = C_INT64_T
#else
  INTEGER, PARAMETER :: ip_ = INT32
  INTEGER, PARAMETER :: ipc_ = C_INT32_T
#endif

END MODULE GALAHAD_KINDS

!-*-*-*-*-  G A L A H A D _  K I N D S _ R E A L   M O D U L E   -*-*-*-*-*-

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
