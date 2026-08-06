! THIS VERSION: GALAHAD 5.3 - 2025-08-31 AT 10:00 GMT

#include "galahad_modules.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  author: Jonathan Hogg
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016
!  A combination of contrib and contrib_free, with a removal of the
!  unnecessary module SSIDS_contrib_fsub_precision, and the transfer of
!  module SSIDS_contrib to GALAHAD_types, GALAHAD 5.3, 2025-08-27

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

! TYPE( contrib_type ), POINTER, VOLATILE :: fcontrib
  TYPE( contrib_type ), POINTER :: fcontrib

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
  USE GALAHAD_SSIDS_numeric_subtree_precision, ONLY: free_contrib
  IMPLICIT NONE

  TYPE( C_PTR ), VALUE :: ccontrib

  TYPE( contrib_type ), POINTER :: fcontrib

  IF ( C_ASSOCIATED( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )

!  only CPU subtrees exist, so cleanup is always the CPU path

    CALL free_contrib( fcontrib )
  END IF
  RETURN

  END SUBROUTINE GALAHAD_SSIDS_contrib_free_precision
