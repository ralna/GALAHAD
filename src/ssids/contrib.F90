! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:10 GMT

#include "spral_procedures.h"
#include "ssids_routines.h"

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  author: Jonathan Hogg
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

MODULE GALAHAD_SSIDS_contrib_precision
  USE GALAHAD_KINDS_precision
  IMPLICIT none

  PRIVATE
  PUBLIC :: contrib_type

!  this type represents a contribution block being passed between two
!  subtrees. It exists in CPU memory, but provides a cleanup routine as
!  memory management may differ between two subtrees being passed.
!  (It would be nice and clean to have a procedure pointer for the cleanup,
!  but alas Fortran/C interop causes severe problems, so we just have the
!  owner value instead and if statements to call the right thing).

  TYPE :: contrib_type
     LOGICAL :: ready = .FALSE.
     INTEGER( ip_ ) :: n ! size of block
     REAL( C_RP_ ), DIMENSION( : ), POINTER :: val ! n x n lwr triangular matrix
     INTEGER( C_IP_ ) :: ldval
     INTEGER( C_IP_ ), DIMENSION( : ), POINTER :: rlist ! row list
     INTEGER( ip_ ) :: ndelay
     INTEGER( C_IP_ ), DIMENSION( : ), POINTER :: delay_perm
     REAL( C_RP_ ), DIMENSION( : ), POINTER :: delay_val
     INTEGER( ip_ ) :: lddelay
     INTEGER( ip_ ) :: owner ! cleanup routine to call: 0=cpu, 1=gpu

!  the following are used by CPU to call correct cleanup routine

     LOGICAL( C_BOOL ) :: posdef
     TYPE( C_PTR ) :: owner_ptr
  END TYPE contrib_type
END MODULE GALAHAD_SSIDS_contrib_precision

!  C function to get interesting components

  SUBROUTINE GALAHAD_SSIDS_contrib_get_data_precision( ccontrib, n, val,       &
                                                       ldval,  rlist, ndelay,  &
                                                       delay_perm, delay_val,  &
                                                       lddelay ) BIND( C )
  USE GALAHAD_KINDS_precision
  USE GALAHAD_SSIDS_contrib_precision
  implicit none

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
! type( contrib_type ), POINTER :: fcontrib

  IF ( C_ASSOCIATED( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )

    DO WHILE ( .NOT. fcontrib%ready )
       ! FIXME: make below a taskyield? ( was: flush )
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
END SUBROUTINE GALAHAD_SSIDS_contrib_get_data_precision
