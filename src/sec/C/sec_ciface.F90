! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S E C    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SEC_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SEC_precision, ONLY:                                           &
        f_sec_control_type   => SEC_control_type,                              &
        f_sec_inform_type    => SEC_inform_type,                               &
        f_sec_full_data_type => SEC_full_data_type,                            &
        f_sec_initialize     => SEC_initialize,                                &
        f_sec_read_specfile  => SEC_read_specfile,                             &
!       f_sec_import         => SEC_import,                                    &
!       f_sec_reset_control  => SEC_reset_control,                             &
        f_sec_information    => SEC_information,                               &
        f_sec_terminate      => SEC_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: sec_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      REAL ( KIND = rpc_ ) :: h_initial
      REAL ( KIND = rpc_ ) :: update_skip_tol
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE sec_control_type

    TYPE, BIND( C ) :: sec_inform_type
      INTEGER ( KIND = ipc_ ) :: status
    END TYPE sec_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( sec_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_sec_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level

    ! Reals
    fcontrol%h_initial = ccontrol%h_initial
    fcontrol%update_skip_tol = ccontrol%update_skip_tol

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_sec_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( sec_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level

    ! Reals
    ccontrol%h_initial = fcontrol%h_initial
    ccontrol%update_skip_tol = fcontrol%update_skip_tol

    ! Strings
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( sec_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_sec_inform_type ), INTENT( OUT ) :: finform

    ! Integers
    finform%status = cinform%status
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_sec_inform_type ), INTENT( IN ) :: finform
    TYPE ( sec_inform_type ), INTENT( OUT ) :: cinform

    ! Integers
    cinform%status = finform%status
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SEC_precision_ciface

!  -------------------------------------
!  C interface to fortran sec_initialize
!  -------------------------------------

  SUBROUTINE sec_initialize( ccontrol, status ) BIND( C )
  USE GALAHAD_SEC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sec_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

! TYPE ( f_sec_full_data_type ), POINTER :: fdata
  TYPE ( f_sec_control_type ) :: fcontrol
  TYPE ( f_sec_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

! ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_sec_initialize( fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
! fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sec_initialize

!  ----------------------------------------
!  C interface to fortran sec_read_specfile
!  ----------------------------------------

  SUBROUTINE sec_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SEC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( sec_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_sec_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_sec_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sec_read_specfile

!  --------------------------------------
!  C interface to fortran sec_information
!  --------------------------------------

  SUBROUTINE sec_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SEC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sec_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_sec_full_data_type ), POINTER :: fdata
  TYPE ( f_sec_inform_type ) :: finform

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

! associate data pointer

! CALL C_F_POINTER( cdata, fdata )

!  obtain SEC solution information

  CALL f_sec_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE sec_information

!  ------------------------------------
!  C interface to fortran sec_terminate
!  ------------------------------------

  SUBROUTINE sec_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SEC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sec_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( sec_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_sec_full_data_type ), pointer :: fdata
  TYPE ( f_sec_control_type ) :: fcontrol
  TYPE ( f_sec_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_sec_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE sec_terminate
