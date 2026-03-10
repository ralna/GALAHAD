! THIS VERSION: GALAHAD 5.5 - 2026-02-19 AT 15:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ R E V E R S E   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 5.5. February 15th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_REVERSE_precision

     USE GALAHAD_KINDS_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SYMBOLS

!  ======================================
!  derived type for reverse communication
!  ======================================

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: REVERSE_type, REVERSE_terminate

!  - - - - - - - - - - -
!   reverse derived type
!  - - - - - - - - - - -

     TYPE :: REVERSE_type
       INTEGER ( KIND = ip_ ) :: index, lvl, lvu, lp
       INTEGER ( KIND = ip_ ) :: eval_status = GALAHAD_ok
       LOGICAL :: transpose
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: iv, ip
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: v, p
     END TYPE REVERSE_type

   CONTAINS

!-*-*-*-*-   R E V E R S E _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE REVERSE_terminate( reverse, status, alloc_status,              &
                                   bad_alloc, out, deallocate_error_fatal )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   reverse - see above
!   status - termination status, 0 = OK, else failure
!   alloc_status - deallocation status
!   bad_alloc - name of failing deallocation array
!   out - unit for error messages
!   deallocate_error_fatal - .TRUE. if deallocation failure is fatal
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( REVERSE_type ), INTENT( INOUT ) :: reverse
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL :: out
     CHARACTER ( LEN = 80 ), INTENT( OUT ), OPTIONAL :: bad_alloc
     LOGICAL, INTENT( IN ), OPTIONAL :: deallocate_error_fatal

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

     array_name = 'slls: reverse%v'
     CALL SPACE_dealloc_array( reverse%v, status, alloc_status,                &
                               array_name = array_name,                        &
                               bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%p'
     CALL SPACE_dealloc_array( reverse%p, status, alloc_status,                &
                               array_name = array_name,                        &
                               bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%iv'
     CALL SPACE_dealloc_array( reverse%iv, status, alloc_status,               &
                               array_name = array_name,                        &
                               bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%ip'
     CALL SPACE_dealloc_array( reverse%ip, status, alloc_status,               &
                               array_name = array_name,                        &
                               bad_alloc = bad_alloc, out = out )
     IF ( deallocate_error_fatal .AND. status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine SLLS_reverse_terminate

     END SUBROUTINE REVERSE_terminate

!  End of module GALAHAD_REVERSE

   END MODULE GALAHAD_REVERSE_precision
