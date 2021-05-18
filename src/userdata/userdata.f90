! THIS VERSION: GALAHAD 3.3 - 18/05/2021 AT 14:00 GMT.

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with as part of NLPT in GALAHAD Version 1.2.
!    November 17th 2002
!   update released with GALAHAD Version 2.0. February 16th 2005
!   separated into its own module, GALAHAD version 3.3, May 18th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_USERDATA_double

      IMPLICIT NONE

      PRIVATE

--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PRIVATE, PARAMETER :: wp = KIND( 1.0D+0 )

!  ======================================
!  The GALAHAD_userdata_type derived type
!  ======================================

      TYPE, PUBLIC :: GALAHAD_userdata_type
         INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer
         REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real
         COMPLEX ( KIND = wcp ), ALLOCATABLE, DIMENSION( : ) :: complex
         CHARACTER, ALLOCATABLE, DIMENSION( : ) :: character
         LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical
         INTEGER, POINTER, DIMENSION( : ) :: integer_pointer => NULL( )
         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: real_pointer => NULL( )
         COMPLEX ( KIND = wcp ), POINTER,                                      &
           DIMENSION( : ) :: complex_pointer => NULL( )
         CHARACTER, POINTER, DIMENSION( : ) :: character_pointer => NULL( )
         LOGICAL, POINTER, DIMENSION( : ) :: logical_pointer => NULL( )
      END TYPE GALAHAD_userdata_type

   END MODULE GALAHAD_USERDATA_double
