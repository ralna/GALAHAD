! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  L M S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_LMS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_LMS_precision, ONLY:                                           &
        f_lms_control_type   => LMS_control_type,                              &
        f_lms_time_type      => LMS_time_type,                                 &
        f_lms_inform_type    => LMS_inform_type,                               &
        f_lms_full_data_type => LMS_full_data_type,                            &
        f_lms_initialize     => LMS_initialize,                                &
        f_lms_read_specfile  => LMS_read_specfile,                             &
        f_lms_information    => LMS_information,                               &
        f_lms_terminate      => LMS_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: lms_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: memory_length
      INTEGER ( KIND = ipc_ ) :: method
      LOGICAL ( KIND = C_BOOL ) :: any_method
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE lms_control_type

    TYPE, BIND( C ) :: lms_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: setup
      REAL ( KIND = rpc_ ) :: form
      REAL ( KIND = rpc_ ) :: apply
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_setup
      REAL ( KIND = rpc_ ) :: clock_form
      REAL ( KIND = rpc_ ) :: clock_apply
    END TYPE lms_time_type

    TYPE, BIND( C ) :: lms_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: length
      LOGICAL ( KIND = C_BOOL ) :: updates_skipped
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( lms_time_type ) :: time
    END TYPE lms_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( lms_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_lms_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%memory_length = ccontrol%memory_length
    fcontrol%method = ccontrol%method

    ! Logicals
    fcontrol%any_method = ccontrol%any_method
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_lms_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( lms_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%memory_length = fcontrol%memory_length
    ccontrol%method = fcontrol%method

    ! Logicals
    ccontrol%any_method = fcontrol%any_method
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Strings
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( lms_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_lms_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%setup = ctime%setup
    ftime%form = ctime%form
    ftime%apply = ctime%apply
    ftime%clock_total = ctime%clock_total
    ftime%clock_setup = ctime%clock_setup
    ftime%clock_form = ctime%clock_form
    ftime%clock_apply = ctime%clock_apply
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_lms_time_type ), INTENT( IN ) :: ftime
    TYPE ( lms_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%setup = ftime%setup
    ctime%form = ftime%form
    ctime%apply = ftime%apply
    ctime%clock_total = ftime%clock_total
    ctime%clock_setup = ftime%clock_setup
    ctime%clock_form = ftime%clock_form
    ctime%clock_apply = ftime%clock_apply
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( lms_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_lms_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%length = cinform%length

    ! Logicals
    finform%updates_skipped = cinform%updates_skipped

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_lms_inform_type ), INTENT( IN ) :: finform
    TYPE ( lms_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%length = finform%length

    ! Logicals
    cinform%updates_skipped = finform%updates_skipped

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_LMS_precision_ciface

!  -------------------------------------
!  C interface to fortran lms_initialize
!  -------------------------------------

  SUBROUTINE lms_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_LMS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( lms_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_lms_full_data_type ), POINTER :: fdata
  TYPE ( f_lms_control_type ) :: fcontrol
  TYPE ( f_lms_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_lms_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE lms_initialize

!  ----------------------------------------
!  C interface to fortran lms_read_specfile
!  ----------------------------------------

  SUBROUTINE lms_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_LMS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( lms_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_lms_control_type ) :: fcontrol
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

  CALL f_lms_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE lms_read_specfile

!  --------------------------------------
!  C interface to fortran lms_information
!  --------------------------------------

  SUBROUTINE lms_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_LMS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( lms_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_lms_full_data_type ), POINTER :: fdata
  TYPE ( f_lms_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain LMS solution information

  CALL f_lms_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE lms_information

!  ------------------------------------
!  C interface to fortran lms_terminate
!  ------------------------------------

  SUBROUTINE lms_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_LMS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( lms_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( lms_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_lms_full_data_type ), pointer :: fdata
  TYPE ( f_lms_control_type ) :: fcontrol
  TYPE ( f_lms_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_lms_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE lms_terminate
