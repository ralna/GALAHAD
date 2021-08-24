! THIS VERSION: GALAHAD 3.3 - 03/05/2021 AT 13:30 GMT.

!-*-*-*-*-*-*-*- G A L A H A D _ U S E R D A T A   M O D U L E -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released as part of NLPT GALAHAD Version 2.0. February 16th 2005
!   seprated into its own module May 3rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_USERDATA_double

     IMPLICIT NONE

     PRIVATE

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PRIVATE, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: wcp = KIND( ( 1.0D+0, 1.0D+0 ) )

!  ======================================
!  The GALAHAD_userdata_type derived type
!  ======================================

     TYPE, PUBLIC :: GALAHAD_userdata_type
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer
       REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real
       COMPLEX ( KIND = wcp ), ALLOCATABLE, DIMENSION( : ) :: complex
       CHARACTER, ALLOCATABLE, DIMENSION( : ) :: character
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical
       INTEGER, POINTER, DIMENSION( : ) :: integer_pointer => null( )
       REAL( KIND = wp ), POINTER, DIMENSION( : ) :: real_pointer => null( )
       COMPLEX ( KIND = wcp ), POINTER,                                        &
         DIMENSION( : ) :: complex_pointer => null( )
       CHARACTER, POINTER, DIMENSION( : ) :: character_pointer => null( )
       LOGICAL, POINTER, DIMENSION( : ) :: logical_pointer => null( )
     END TYPE GALAHAD_userdata_type

!  End of module USERDATA

   END MODULE GALAHAD_USERDATA_double

