! THIS VERSION: GALAHAD 4.1 - 2022-12-06 AT 07:15 GMT.

!-*-*-*-*-*-*-*- G A L A H A D _ P R E C I S I O N   M O D U L E -*-*-*-*-*-*-*-

!  Copyright reserved, GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released, GALAHAD Version 4.1, December 6th 2022

MODULE GALAHAD_precision
  USE ISO_C_BINDING
  USE ISO_FORTRAN_ENV

  IMPLICIT NONE

  PUBLIC

!---------------------------------------------
!   R e a l  a n d   i n t e g e r   t y p e s
!---------------------------------------------

!  basic kinds used

 INTEGER, PARAMETER :: i4 = INT32
 INTEGER, PARAMETER :: i8 = INT64
 INTEGER, PARAMETER :: r4 = REAL32
 INTEGER, PARAMETER :: r8 = REAL64
 INTEGER, PARAMETER :: c4 = KIND( ( 1.0_r4, 1.0_r4 ) )
 INTEGER, PARAMETER :: c8 = KIND( ( 1.0_r8, 1.0_r8 ) )

!  common aliases

 INTEGER, PARAMETER :: sp = r4
 INTEGER, PARAMETER :: dp = r8
 INTEGER, PARAMETER :: long = i8

!--------------------------------
!   P r e c i s i o n s  u s e d
!--------------------------------

#ifdef GALAHAD_SINGLE
 INTEGER, PARAMETER :: wp = r4
 INTEGER, PARAMETER :: cp = c4
 INTEGER, PARAMETER :: wpc = C_FLOAT
#else
 INTEGER, PARAMETER :: wp = r8
 INTEGER, PARAMETER :: cp = c8
 INTEGER, PARAMETER :: wpc = C_DOUBLE
#endif

#ifdef GALAHAD_64BIT_INTEGER
 INTEGER, PARAMETER :: ip = i8
 INTEGER, PARAMETER :: ipc = C_INT64_T
#else
 INTEGER, PARAMETER :: ip = i4
 INTEGER, PARAMETER :: ipc = C_INT32_T
#endif

END MODULE GALAHAD_precision
