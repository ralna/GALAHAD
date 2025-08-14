! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:20 GMT

#include "spral_procedures.h"

!  copyright 2016 The Science and Technology Facilities Council (STFC)
!  licence   BSD licence, see LICENCE file for details
!  author    Jonathan Hogg
!
!  Routines for freeing contrib_type.
!
!  As it depends on routines defined by module that use the type, it needs
!  to be a seperate module to GALAHAD_SSIDS_CONTRIB_PRECISION.

MODULE GALAHAD_SSIDS_CONTRIB_FSUB_precision
  USE SPRAL_KINDS_precision
  USE GALAHAD_SSIDS_CONTRIB_precision, ONLY: contrib_type
  USE GALAHAD_SSIDS_CPU_SUBTREE_precision, ONLY: cpu_free_contrib
  USE GALAHAD_SSIDS_GPU_SUBTREE_precision, ONLY: gpu_free_contrib
  IMPLICIT none

CONTAINS
  SUBROUTINE contrib_free(contrib)
    IMPLICIT none
    TYPE( contrib_type ), INTENT( INOUT ) :: contrib

    SELECT CASE(contrib%owner)
    CASE (0) ! CPU
       CALL cpu_free_contrib( contrib%posdef, contrib%owner_ptr )
    CASE (1) ! GPU
       CALL gpu_free_contrib( contrib )
    CASE DEFAULT
       ! This should never happen
       PRINT *, "Unrecognised contrib owner ", contrib%owner
       STOP -1
    END SELECT
  END SUBROUTINE contrib_free
END MODULE galahad_ssids_contrib_fsub_precision

! The C prototype for the following routine is in contrib.h

SUBROUTINE galahad_ssids_contrib_free_precision( ccontrib ) BIND(C)
  USE, INTRINSIC :: iso_c_binding
  USE GALAHAD_SSIDS_CONTRIB_FSUB_precision
  IMPLICIT NONE

  TYPE( C_PTR ), VALUE :: ccontrib

  TYPE( contrib_type ), POINTER :: fcontrib

  IF ( C_associated( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )
    CALL contrib_free( fcontrib )
  END IF
END SUBROUTINE galahad_ssids_contrib_free_precision
