! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:10 GMT

#include "spral_procedures.h"
#include "ssids_routines.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg

MODULE GALAHAD_SSIDS_CONTRIB_precision
  USE SPRAL_KINDS_precision
  IMPLICIT none

  PRIVATE
  PUBLIC :: contrib_type

  ! This type represents a contribution block being passed between two
  ! subtrees. It exists in CPU memory, but provides a cleanup routine as
  ! memory management may differ between two subtrees being passed.
  ! (It would be nice and clean to have a procedure pointer for the cleanup,
  ! but alas Fortran/C interop causes severe problems, so we just have the
  ! owner value instead and if statements to call the right thing).

  TYPE :: contrib_type
     LOGICAL :: ready = .false.
     INTEGER( ip_ ) :: n ! size of block
     REAL( C_RP_ ), dimension( : ), pointer :: val ! n x n lwr triangular matrix
     INTEGER( C_IP_ ) :: ldval
     INTEGER( C_IP_ ), dimension( : ), pointer :: rlist ! row list
     INTEGER( ip_ ) :: ndelay
     INTEGER( C_IP_ ), dimension( : ), pointer :: delay_perm
     REAL( C_RP_ ), dimension( : ), pointer :: delay_val
     INTEGER( ip_ ) :: lddelay
     INTEGER( ip_ ) :: owner ! cleanup routine to call: 0=cpu, 1=gpu
     ! Following are used by CPU to call correct cleanup routine
     LOGICAL( C_BOOL ) :: posdef
     TYPE( C_PTR ) :: owner_ptr
  END TYPE contrib_type
END MODULE GALAHAD_SSIDS_CONTRIB_precision

!  C function to get interesting components

SUBROUTINE galahad_ssids_contrib_get_data_precision( ccontrib, n, val, ldval,  &
                       rlist, ndelay, delay_perm, delay_val, lddelay ) BIND( C )
  USE SPRAL_KINDS_precision
  USE GALAHAD_SSIDS_CONTRIB_precision
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

  TYPE( contrib_type ), pointer, volatile :: fcontrib
! type( contrib_type ), pointer :: fcontrib

  IF ( c_associated( ccontrib ) ) THEN
    CALL C_F_POINTER( ccontrib, fcontrib )

    DO WHILE ( .NOT. fcontrib%ready )
       ! FIXME: make below a taskyield? ( was: flush )
       !$omp taskyield
    END DO

    n = fcontrib%n
    val = c_loc( fcontrib%val )
    ldval = fcontrib%ldval
    rlist = c_loc( fcontrib%rlist )
    ndelay = fcontrib%ndelay
    IF ( ASSOCIATED( fcontrib%delay_val ) ) THEN
      delay_perm = c_loc( fcontrib%delay_perm )
      delay_val = c_loc( fcontrib%delay_val )
    ELSE
      delay_perm = c_null_ptr
      delay_val = c_null_ptr
    END IF
    lddelay = fcontrib%lddelay
  END IF
END SUBROUTINE galahad_ssids_contrib_get_data_precision
