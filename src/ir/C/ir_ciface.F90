! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  I R    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.4. January 4th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_IR_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_IR_precision, ONLY:                                            &
        f_ir_control_type   => IR_control_type,                                &
        f_ir_inform_type    => IR_inform_type,                                 &
        f_ir_full_data_type => IR_full_data_type,                              &
        f_ir_initialize     => IR_initialize,                                  &
        f_ir_read_specfile  => IR_read_specfile,                               &
        f_ir_information    => IR_information,                                 &
        f_ir_terminate      => IR_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ir_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: itref_max
      REAL ( KIND = rpc_ ) :: acceptable_residual_relative
      REAL ( KIND = rpc_ ) :: acceptable_residual_absolute
      REAL ( KIND = rpc_ ) :: required_residual_relative
      LOGICAL ( KIND = C_BOOL ) :: record_residuals
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE ir_control_type

    TYPE, BIND( C ) :: ir_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      REAL ( KIND = rpc_ ) :: norm_initial_residual
      REAL ( KIND = rpc_ ) :: norm_final_residual
    END TYPE ir_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( ir_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ir_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%itref_max = ccontrol%itref_max

    ! Reals
    fcontrol%acceptable_residual_relative                                      &
      = ccontrol%acceptable_residual_relative
    fcontrol%acceptable_residual_absolute                                      &
      = ccontrol%acceptable_residual_absolute
    fcontrol%required_residual_relative                                        &
      = ccontrol%required_residual_relative

    ! Logicals
    fcontrol%record_residuals = ccontrol%record_residuals
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
    TYPE ( f_ir_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( ir_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%itref_max = fcontrol%itref_max

    ! Reals
    ccontrol%acceptable_residual_relative                                      &
      = fcontrol%acceptable_residual_relative
    ccontrol%acceptable_residual_absolute                                      &
      = fcontrol%acceptable_residual_absolute
    ccontrol%required_residual_relative                                        &
      = fcontrol%required_residual_relative

    ! Logicals
    ccontrol%record_residuals = fcontrol%record_residuals
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
    TYPE ( ir_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_ir_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status

    ! Reals
    finform%norm_initial_residual = cinform%norm_initial_residual
    finform%norm_final_residual = cinform%norm_final_residual

    ! Strings
     DO i = 1, LEN( finform%bad_alloc )
       IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
       finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
     END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_ir_inform_type ), INTENT( IN ) :: finform
    TYPE ( ir_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status

    ! Reals
    cinform%norm_initial_residual = finform%norm_initial_residual
    cinform%norm_final_residual = finform%norm_final_residual

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
       cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_IR_precision_ciface

!  -------------------------------------
!  C interface to fortran ir_initialize
!  -------------------------------------

  SUBROUTINE ir_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_IR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( ir_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_ir_full_data_type ), POINTER :: fdata
  TYPE ( f_ir_control_type ) :: fcontrol
  TYPE ( f_ir_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_ir_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE ir_initialize

!  ----------------------------------------
!  C interface to fortran ir_read_specfile
!  ----------------------------------------

  SUBROUTINE ir_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_IR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( ir_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_ir_control_type ) :: fcontrol
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

  CALL f_ir_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE ir_read_specfile

!  --------------------------------------
!  C interface to fortran ir_information
!  --------------------------------------

  SUBROUTINE ir_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_IR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ir_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_ir_full_data_type ), pointer :: fdata
  TYPE ( f_ir_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain IR solution information

  CALL f_ir_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE ir_information

!  ------------------------------------
!  C interface to fortran ir_terminate
!  ------------------------------------

  SUBROUTINE ir_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_IR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ir_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( ir_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_ir_full_data_type ), pointer :: fdata
  TYPE ( f_ir_control_type ) :: fcontrol
  TYPE ( f_ir_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_ir_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE ir_terminate
