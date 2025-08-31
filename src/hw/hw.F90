! THIS VERSION: GALAHAD 5.3 - 2025-08-31 AT 10:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ H W   M O D U L E  *-*-*-*-*-*-*-*-*-*-*-

!    --------------------------------------------------------------------
!   | Hardware toplology package originally spral_hw_topology from SPRAL |
!    --------------------------------------------------------------------

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

      MODULE GALAHAD_HW

!  provides routines for detecting and/or specifying hardware topology for 
!  topology-aware routines

        USE GALAHAD_KINDS, ONLY: ip_, c_ip_
        USE, INTRINSIC :: iso_c_binding
        IMPLICIT NONE

        PRIVATE
        PUBLIC :: HW_numa_region, HW_c_numa_region, HW_guess_topology

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  derived type describing regions

        TYPE :: HW_numa_region

!  number of processors in region

          INTEGER ( KIND = ip_ ) :: nproc

!  list of attached GPUs

          INTEGER ( KIND = ip_ ), DIMENSION ( : ), ALLOCATABLE :: gpus 
        END TYPE HW_numa_region

!  fortran interoperable definition of galahad::hw_topology::NumaRegion

        TYPE, BIND( C ) :: HW_c_numa_region
          INTEGER ( KIND = c_int ) :: nproc
          INTEGER ( KIND = c_int ) :: ngpu
          TYPE ( c_ptr ) :: gpus
        END TYPE HW_c_numa_region

!----------------------
!   I n t e r f a c e s
!----------------------

!  fortran interfaces to C procedures

        INTERFACE
          SUBROUTINE galahad_hw_topology_guess( nregion, regions ) BIND( C )
            USE, INTRINSIC :: iso_c_binding
            IMPLICIT NONE
            INTEGER ( KIND = c_int ), INTENT ( OUT ) :: nregion
            TYPE ( c_ptr ), INTENT ( OUT ) :: regions
          END SUBROUTINE galahad_hw_topology_guess

          SUBROUTINE galahad_hw_topology_free( nregion, regions ) BIND( C )
            USE, INTRINSIC :: iso_c_binding
            IMPLICIT NONE
            INTEGER ( KIND = c_int ), VALUE :: nregion
            TYPE ( c_ptr ), VALUE :: regions
          END SUBROUTINE  galahad_hw_topology_free
        END INTERFACE

      CONTAINS

!-*-*-*-*-   H W _ G U E S S _ T O P O L O G Y   S U B R O U T I N E   -*-*-*-*

        SUBROUTINE HW_guess_topology( regions, st )

!  return best guess for machine topology

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

!  upon return allocated to have size equal to the number of NUMA regions. 
!  The members describe each region

        TYPE ( HW_numa_region ), DIMENSION( : ), ALLOCATABLE,                  &
                                                 INTENT( OUT ) :: regions

!  status return from allocate. If non-zero upon return, an allocation failed

        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: st

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

        INTEGER ( KIND = c_int ) :: i
        INTEGER ( KIND = c_int ) :: nregions
        TYPE ( c_ptr ) :: c_regions
        TYPE ( HW_c_numa_region ), DIMENSION ( : ), POINTER,                   &
                                                    CONTIGUOUS :: f_regions
        INTEGER ( KIND = c_ip_ ), DIMENSION ( : ), POINTER,                    &
                                                   CONTIGUOUS :: f_gpus

!  get regions from C

        CALL galahad_hw_topology_guess( nregions, c_regions )
        IF ( c_associated( c_regions ) ) THEN
          CALL c_f_pointer( c_regions, f_regions, shape = (/ nregions /) )

!  copy to allocatable array

          ALLOCATE ( regions( nregions ), STAT = st )
          IF ( st /= 0 ) RETURN
          DO i = 1, nregions
            regions( i )%nproc = f_regions( i )%nproc
            ALLOCATE ( regions( i )%gpus( f_regions( i )%ngpu ), STAT = st )
            IF ( st /= 0 ) RETURN
            IF ( f_regions( i )%ngpu > 0 ) THEN
              CALL c_f_pointer( f_regions( i )%gpus, f_gpus,                   &
                                shape = (/ f_regions( i )%ngpu /)  )
              regions( i )%gpus = f_gpus( : )
            END IF
          END DO
        END IF

!  free C version

        CALL galahad_hw_topology_free( nregions, c_regions )
        RETURN

        END SUBROUTINE HW_guess_topology

    END MODULE GALAHAD_HW
