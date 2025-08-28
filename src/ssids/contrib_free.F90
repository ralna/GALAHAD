! THIS VERSION: GALAHAD 5.3 - 2025-08-27 AT 15:40 GMT

#include "ssids_procedures.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  author: Jonathan Hogg
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016
!  Remove unnecessary module, GALAHAD 5.3, 2025-08-27

!!$  MODULE GALAHAD_SSIDS_contrib_fsub_precision
!!$
!!$!  routines for freeing contrib_type
!!$!
!!$!  as the module depends on routines defined by module that use the type, 
!!$!  it needs to be a seperate module to GALAHAD_SSIDS_contrib_precision
!!$
!!$    USE GALAHAD_KINDS_precision
!!$    IMPLICIT NONE
!!$
!!$  CONTAINS
!!$
!!$!-  G A L A H A D -  S S I D S _ c o n t r i b _f s u b  S U B R O U T I N E 
!!$
!!$    SUBROUTINE contrib_free( fcontrib )
!!$    IMPLICIT none
!!$    TYPE( contrib_type ), INTENT( INOUT ) :: fcontrib
!!$    SELECT CASE( contrib%owner )
!!$    CASE ( 0 ) ! CPU
!!$       CALL cpu_free_contrib( contrib%posdef, contrib%owner_ptr )
!!$    CASE ( 1 ) ! GPU
!!$       CALL gpu_free_contrib( contrib )
!!$    CASE DEFAULT ! This should never happen
!!$       PRINT *, "Unrecognised contrib owner ", contrib%owner
!!$       STOP - 1
!!$    END SELECT
!!$    RETURN
!!$
!!$    END SUBROUTINE contrib_free
!!$  END MODULE GALAHAD_SSIDS_contrib_fsub_precision

!-  G A L A H A D -  S S I D S _ c o n t r i b _f r e e  S U B R O U T I N E  -

  SUBROUTINE GALAHAD_SSIDS_contrib_free_precision( ccontrib ) BIND( C )

!  the C prototype for the following routine is in contrib.h

  USE, INTRINSIC :: iso_c_binding
  USE GALAHAD_SSIDS_contrib_precision, ONLY: contrib_type
  USE GALAHAD_SSIDS_cpu_subtree_precision, ONLY: cpu_free_contrib
  USE GALAHAD_SSIDS_gpu_subtree_precision, ONLY: gpu_free_contrib
! USE GALAHAD_SSIDS_contrib_fsub_precision
  IMPLICIT NONE

  TYPE( C_PTR ), VALUE :: ccontrib

  TYPE( contrib_type ), POINTER :: fcontrib

  IF ( C_ASSOCIATED( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )
    SELECT CASE( fcontrib%owner )
    CASE ( 0 ) ! CPU
       CALL cpu_free_contrib( fcontrib%posdef, fcontrib%owner_ptr )
    CASE ( 1 ) ! GPU
       CALL gpu_free_contrib( fcontrib )
    CASE DEFAULT ! This should never happen
       PRINT *, "Unrecognised contrib owner ", fcontrib%owner
       STOP - 1
    END SELECT
!   CALL contrib_free( fcontrib )
  END IF
  END SUBROUTINE GALAHAD_SSIDS_contrib_free_precision
