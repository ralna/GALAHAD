! THIS VERSION: GALAHAD 4.3 - 2024-01-16 AT 10:30 GMT.

#ifdef SPRAL_64BIT_INTEGER
#define SPRAL_KINDS_double spral_kinds_double_64
#define SPRAL_KINDS_single spral_kinds_single_64
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

!  common aliases

  INTEGER, PARAMETER :: sp_ = r4_
  INTEGER, PARAMETER :: dp_ = r8_
  INTEGER, PARAMETER :: long_ = i8_
  INTEGER, PARAMETER :: spc_ = C_FLOAT
  INTEGER, PARAMETER :: dpc_ = C_DOUBLE
  INTEGER, PARAMETER :: longc_ = C_INT64_T

!  integer kinds

#ifdef SPRAL_64BIT_INTEGER
  INTEGER, PARAMETER :: ip_ = INT64
  INTEGER, PARAMETER :: ipc_ = C_INT64_T
#else
  INTEGER, PARAMETER :: ip_ = INT32
  INTEGER, PARAMETER :: ipc_ = C_INT32_T
#endif

  INTEGER, PARAMETER :: C_IP_ = ipc_
  INTEGER, PARAMETER :: CLONG_ = longc_

END MODULE SPRAL_KINDS

!-*-*-*-*-  S P R A L _  K I N D S _ S I N G L E  M O D U L E   -*-*-*-*-*-

MODULE SPRAL_KINDS_single
  USE SPRAL_KINDS
  IMPLICIT NONE
  PUBLIC

!--------------------------------------------------------
!  R e a l  k i n d s  ( s i n g l e  p r e c i s i o n )
!--------------------------------------------------------

  INTEGER, PARAMETER :: real_bytes_ = 4
  INTEGER, PARAMETER :: rp_ = r4_
  INTEGER, PARAMETER :: cp_ = c4_
  INTEGER, PARAMETER :: rpc_ = spc_
  INTEGER, PARAMETER :: C_RP_ = rpc_

END MODULE SPRAL_KINDS_single

!-*-*-*-*-  S P R A L _  K I N D S _ D O U B L E  M O D U L E   -*-*-*-*-*-

MODULE SPRAL_KINDS_double
  USE SPRAL_KINDS
  IMPLICIT NONE
  PUBLIC

!--------------------------------------------------------
!  R e a l  k i n d s  ( d o u b l e  p r e c i s i o n )
!--------------------------------------------------------

  INTEGER, PARAMETER :: real_bytes_ = 8
  INTEGER, PARAMETER :: rp_ = r8_
  INTEGER, PARAMETER :: cp_ = c8_
  INTEGER, PARAMETER :: rpc_ = dpc_
  INTEGER, PARAMETER :: C_RP_ = rpc_

END MODULE SPRAL_KINDS_double
