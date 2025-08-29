! THIS VERSION: GALAHAD 5.3 - 2025-08-29 AT 12:30 GMT

#include "ssids_procedures.h"
#include "ssids_routines.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  author: Jonathan Hogg
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016
!  A combination of contrib and contrib_free, with a removal of the
!  unnecessary module SSIDS_contrib_fsub_precision, and the transfer of
!  module SSIDS_contrib to GALAHAD_types, GALAHAD 5.3, 2025-08-27

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

! G A L A H A D - S S I D S _ c o n t r i b _g e t _d a t a  S U B R O U T I N E

  SUBROUTINE GALAHAD_SSIDS_contrib_get_data_precision( ccontrib, n, val,       &
                                                       ldval,  rlist, ndelay,  &
                                                       delay_perm, delay_val,  &
                                                       lddelay ) BIND( C )

!  C function to get interesting components of the contrib type

  USE GALAHAD_KINDS_precision
  USE GALAHAD_SSIDS_types_precision, ONLY: contrib_type
  IMPLICIT NONE

  TYPE( C_PTR ), VALUE :: ccontrib
  INTEGER( C_IP_ ), INTENT( OUT ) :: n
  TYPE( C_PTR ), INTENT( OUT ) :: val
  INTEGER( C_IP_ ), INTENT( OUT ) :: ldval
  TYPE( C_PTR ), INTENT( OUT ) :: rlist
  INTEGER( C_IP_ ), INTENT( OUT ) :: ndelay
  TYPE( C_PTR ), INTENT( OUT ) :: delay_perm
  TYPE( C_PTR ), INTENT( OUT ) :: delay_val
  INTEGER( C_IP_ ), INTENT( OUT ) :: lddelay

  TYPE( contrib_type ), POINTER, VOLATILE :: fcontrib

  IF ( C_ASSOCIATED( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )

    DO WHILE ( .NOT. fcontrib%ready )
      ! FIXME: make below a taskyield? (was: flush)
!$omp taskyield
    END DO

    n = fcontrib%n
    val = C_LOC( fcontrib%val )
    ldval = fcontrib%ldval
    rlist = C_LOC( fcontrib%rlist )
    ndelay = fcontrib%ndelay
    IF ( ASSOCIATED( fcontrib%delay_val ) ) THEN
      delay_perm = C_LOC( fcontrib%delay_perm )
      delay_val = C_LOC( fcontrib%delay_val )
    ELSE
      delay_perm = C_NULL_PTR
      delay_val = C_NULL_PTR
    END IF
    lddelay = fcontrib%lddelay
  END IF
  RETURN

  END SUBROUTINE GALAHAD_SSIDS_contrib_get_data_precision

!-  G A L A H A D -  S S I D S _ c o n t r i b _f r e e  S U B R O U T I N E  -

  SUBROUTINE GALAHAD_SSIDS_contrib_free_precision( ccontrib ) BIND( C )

!  the C prototype for the following routine is in contrib.h

  USE, INTRINSIC :: iso_c_binding
  USE GALAHAD_SSIDS_types_precision, ONLY: contrib_type
  USE GALAHAD_SSIDS_cpu_subtree_precision, ONLY: cpu_free_contrib
! USE GALAHAD_SSIDS_gpu_subtree_precision, ONLY: gpu_free_contrib
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
!      CALL gpu_free_contrib( fcontrib ) 
       fcontrib%n = 0
    CASE DEFAULT ! This should never happen
       PRINT *, "Unrecognised contrib owner ", fcontrib%owner
       STOP - 1
    END SELECT
!   CALL contrib_free( fcontrib )
  END IF
  RETURN

  END SUBROUTINE GALAHAD_SSIDS_contrib_free_precision
