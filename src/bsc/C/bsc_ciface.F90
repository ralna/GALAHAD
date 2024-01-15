! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  B S C    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_BSC_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_BSC_precision, ONLY:                                           &
        f_bsc_control_type => BSC_control_type,                                &
        f_bsc_inform_type => BSC_inform_type,                                  &
        f_bsc_full_data_type => BSC_full_data_type,                            &
        f_bsc_initialize => BSC_initialize,                                    &
        f_bsc_read_specfile => BSC_read_specfile,                              &
!       f_bsc_import => BSC_import,                                            &
!       f_bsc_reset_control => BSC_reset_control,                              &
        f_bsc_information => BSC_information,                                  &
        f_bsc_terminate => BSC_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: bsc_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: max_col
      INTEGER ( KIND = ipc_ ) :: new_a
      INTEGER ( KIND = ipc_ ) :: extra_space_s
      LOGICAL ( KIND = C_BOOL ) :: s_also_by_column
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE bsc_control_type

    TYPE, BIND( C ) :: bsc_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: max_col_a
      INTEGER ( KIND = ipc_ ) :: exceeds_max_col
      REAL ( KIND = rpc_ ) :: time
      REAL ( KIND = rpc_ ) :: clock_time
    END TYPE bsc_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( bsc_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_bsc_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%max_col = ccontrol%max_col
    fcontrol%new_a = ccontrol%new_a
    fcontrol%extra_space_s = ccontrol%extra_space_s

    ! Logicals
    fcontrol%s_also_by_column = ccontrol%s_also_by_column
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
    TYPE ( f_bsc_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( bsc_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%max_col = fcontrol%max_col
    ccontrol%new_a = fcontrol%new_a
    ccontrol%extra_space_s = fcontrol%extra_space_s

    ! Logicals
    ccontrol%s_also_by_column = fcontrol%s_also_by_column
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

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( bsc_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_bsc_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%max_col_a = cinform%max_col_a
    finform%exceeds_max_col = cinform%exceeds_max_col

    ! Reals
    finform%time = cinform%time
    finform%clock_time = cinform%clock_time

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_bsc_inform_type ), INTENT( IN ) :: finform
    TYPE ( bsc_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%max_col_a = finform%max_col_a
    cinform%exceeds_max_col = finform%exceeds_max_col

    ! Reals
    cinform%time = finform%time
    cinform%clock_time = finform%clock_time

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_BSC_precision_ciface

!  -------------------------------------
!  C interface to fortran bsc_initialize
!  -------------------------------------

  SUBROUTINE bsc_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_BSC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( bsc_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_bsc_full_data_type ), POINTER :: fdata
  TYPE ( f_bsc_control_type ) :: fcontrol
  TYPE ( f_bsc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_bsc_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bsc_initialize

!  ----------------------------------------
!  C interface to fortran bsc_read_specfile
!  ----------------------------------------

  SUBROUTINE bsc_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_BSC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( bsc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_bsc_control_type ) :: fcontrol
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

  CALL f_bsc_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bsc_read_specfile

!  --------------------------------------
!  C interface to fortran bsc_information
!  --------------------------------------

  SUBROUTINE bsc_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_BSC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( bsc_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_bsc_full_data_type ), pointer :: fdata
  TYPE ( f_bsc_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain BSC solution information

  CALL f_bsc_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE bsc_information

!  ------------------------------------
!  C interface to fortran bsc_terminate
!  ------------------------------------

  SUBROUTINE bsc_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_BSC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( bsc_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( bsc_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_bsc_full_data_type ), pointer :: fdata
  TYPE ( f_bsc_control_type ) :: fcontrol
  TYPE ( f_bsc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_bsc_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE bsc_terminate
