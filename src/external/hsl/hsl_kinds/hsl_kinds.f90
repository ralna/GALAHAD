! THIS VERSION: HSL SUBSET 1.1 - 2024-11-23 AT 15:20 GMT

#include "hsl_subset.h"

!-*-*-*-*-*-*-  H S L _ S U B S E T _ K I N D S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, HSL subset
!  Principal author: Nick Gould

!  History -
!   originally released, HSL subset Version 1.0, February 14th 2024

MODULE HSL_KINDS
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

END MODULE HSL_KINDS

!-*-*-*-  H S L _ S U B S E T _  K I N D S _ S I N G L E  M O D U L E   -*-*-*-

MODULE HSL_KINDS_single
  USE HSL_KINDS
  IMPLICIT NONE
  PUBLIC

!--------------------------------------------------------
!  R e a l  k i n d s  ( s i n g l e  p r e c i s i o n )
!--------------------------------------------------------

  INTEGER, PARAMETER :: real_bytes_ = 4
  INTEGER, PARAMETER :: rp_ = r4_
  INTEGER, PARAMETER :: cp_ = c4_
  INTEGER, PARAMETER :: rpc_ = spc_

END MODULE HSL_KINDS_single

!-*-*-*-  H S L _ S U B S E T _  K I N D S _ D O U B L E  M O D U L E   -*-*-*-

MODULE HSL_KINDS_double
  USE HSL_KINDS
  IMPLICIT NONE
  PUBLIC

!--------------------------------------------------------
!  R e a l  k i n d s  ( d o u b l e  p r e c i s i o n )
!--------------------------------------------------------

  INTEGER, PARAMETER :: real_bytes_ = 8
  INTEGER, PARAMETER :: rp_ = r8_
  INTEGER, PARAMETER :: cp_ = c8_
  INTEGER, PARAMETER :: rpc_ = dpc_

END MODULE HSL_KINDS_double

#ifdef REAL_128
!-*-  H S L _ S U B S E T _  K I N D S _ Q U A D R U P L E  M O D U L E   -*-

MODULE HSL_KINDS_quadruple
  USE HSL_KINDS
  IMPLICIT NONE
  PUBLIC

!--------------------------------------------------------------
!  R e a l  k i n d s  ( q u a d r u p l e  p r e c i s i o n )
!--------------------------------------------------------------

  INTEGER, PARAMETER :: real_bytes_ = 16
  INTEGER, PARAMETER :: rp_ = r16_
  INTEGER, PARAMETER :: cp_ = c16_
  INTEGER, PARAMETER :: rpc_ = qpc_

END MODULE HSL_KINDS_quadruple
#endif
