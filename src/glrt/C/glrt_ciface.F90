! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  G L R T    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. December 16th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_GLRT_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_GLRT_precision, ONLY:                                          &
        f_glrt_control_type => GLRT_control_type,                              &
        f_glrt_inform_type => GLRT_inform_type,                                &
        f_glrt_full_data_type => GLRT_full_data_type,                          &
        f_glrt_initialize => GLRT_initialize,                                  &
        f_glrt_read_specfile => GLRT_read_specfile,                            &
        f_glrt_import_control => GLRT_import_control,                          &
        f_glrt_solve_problem => GLRT_solve_problem,                            &
        f_glrt_information => GLRT_information,                                &
        f_glrt_terminate => GLRT_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: glrt_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: itmax
      INTEGER ( KIND = ipc_ ) :: stopping_rule
      INTEGER ( KIND = ipc_ ) :: freq
      INTEGER ( KIND = ipc_ ) :: extra_vectors
      INTEGER ( KIND = ipc_ ) :: ritz_printout_device
      REAL ( KIND = rpc_ ) :: stop_relative
      REAL ( KIND = rpc_ ) :: stop_absolute
      REAL ( KIND = rpc_ ) :: fraction_opt
      REAL ( KIND = rpc_ ) :: rminvr_zero
      REAL ( KIND = rpc_ ) :: f_0
      LOGICAL ( KIND = C_BOOL ) :: unitm
      LOGICAL ( KIND = C_BOOL ) :: impose_descent
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: print_ritz_values
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: ritz_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE glrt_control_type

    TYPE, BIND( C ) :: glrt_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: iter_pass2
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: obj_regularized
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: xpo_norm
      REAL ( KIND = rpc_ ) :: leftmost
      LOGICAL ( KIND = C_BOOL ) :: negative_curvature
      LOGICAL ( KIND = C_BOOL ) :: hard_case
    END TYPE glrt_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( glrt_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_glrt_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%itmax = ccontrol%itmax
    fcontrol%stopping_rule = ccontrol%stopping_rule
    fcontrol%freq = ccontrol%freq
    fcontrol%extra_vectors = ccontrol%extra_vectors
    fcontrol%ritz_printout_device = ccontrol%ritz_printout_device

    ! Reals
    fcontrol%stop_relative = ccontrol%stop_relative
    fcontrol%stop_absolute = ccontrol%stop_absolute
    fcontrol%fraction_opt = ccontrol%fraction_opt
    fcontrol%rminvr_zero = ccontrol%rminvr_zero
    fcontrol%f_0 = ccontrol%f_0

    ! Logicals
    fcontrol%unitm = ccontrol%unitm
    fcontrol%impose_descent = ccontrol%impose_descent
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%print_ritz_values = ccontrol%print_ritz_values

    ! Strings
    DO i = 1, LEN( fcontrol%ritz_file_name )
      IF ( ccontrol%ritz_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%ritz_file_name( i : i ) = ccontrol%ritz_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_glrt_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( glrt_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%itmax = fcontrol%itmax
    ccontrol%stopping_rule = fcontrol%stopping_rule
    ccontrol%freq = fcontrol%freq
    ccontrol%extra_vectors = fcontrol%extra_vectors
    ccontrol%ritz_printout_device = fcontrol%ritz_printout_device

    ! Reals
    ccontrol%stop_relative = fcontrol%stop_relative
    ccontrol%stop_absolute = fcontrol%stop_absolute
    ccontrol%fraction_opt = fcontrol%fraction_opt
    ccontrol%rminvr_zero = fcontrol%rminvr_zero
    ccontrol%f_0 = fcontrol%f_0

    ! Logicals
    ccontrol%unitm = fcontrol%unitm
    ccontrol%impose_descent = fcontrol%impose_descent
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%print_ritz_values = fcontrol%print_ritz_values

    ! Strings
    l = LEN( fcontrol%ritz_file_name )
    DO i = 1, l
      ccontrol%ritz_file_name( i ) = fcontrol%ritz_file_name( i : i )
    END DO
    ccontrol%ritz_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( glrt_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_glrt_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%iter_pass2 = cinform%iter_pass2

    ! Reals
    finform%obj = cinform%obj
    finform%obj_regularized = cinform%obj_regularized
    finform%multiplier = cinform%multiplier
    finform%xpo_norm = cinform%xpo_norm
    finform%leftmost = cinform%leftmost

    ! Logicals
    finform%negative_curvature = cinform%negative_curvature
    finform%hard_case = cinform%hard_case

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_glrt_inform_type ), INTENT( IN ) :: finform
    TYPE ( glrt_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%iter_pass2 = finform%iter_pass2

    ! Reals
    cinform%obj = finform%obj
    cinform%obj_regularized = finform%obj_regularized
    cinform%multiplier = finform%multiplier
    cinform%xpo_norm = finform%xpo_norm
    cinform%leftmost = finform%leftmost

    ! Logicals
    cinform%negative_curvature = finform%negative_curvature
    cinform%hard_case = finform%hard_case

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_GLRT_precision_ciface

!  -------------------------------------
!  C interface to fortran glrt_initialize
!  -------------------------------------

  SUBROUTINE glrt_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( glrt_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_glrt_full_data_type ), POINTER :: fdata
  TYPE ( f_glrt_control_type ) :: fcontrol
  TYPE ( f_glrt_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_glrt_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE glrt_initialize

!  ----------------------------------------
!  C interface to fortran glrt_read_specfile
!  ----------------------------------------

  SUBROUTINE glrt_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( glrt_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_glrt_control_type ) :: fcontrol
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

  CALL f_glrt_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE glrt_read_specfile

!  ------------------------------------------
!  C interface to fortran glrt_import_control
!  ------------------------------------------

  SUBROUTINE glrt_import_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( glrt_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_glrt_control_type ) :: fcontrol
  TYPE ( f_glrt_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required GLRT structure

  CALL f_glrt_import_control( fcontrol, fdata, status )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE glrt_import_control

!  -----------------------------------------
!  C interface to fortran glrt_solve_problem
!  ----------------------------------------_

  SUBROUTINE glrt_solve_problem( cdata, status, n,power, weight, x, r,         &
                                 vector ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: power, weight
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: r
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: vector
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_glrt_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_glrt_solve_problem( fdata, status, n, power, weight, x, r, vector )
  RETURN

  END SUBROUTINE glrt_solve_problem

!  --------------------------------------
!  C interface to fortran glrt_information
!  --------------------------------------

  SUBROUTINE glrt_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( glrt_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_glrt_full_data_type ), pointer :: fdata
  TYPE ( f_glrt_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain GLRT solution information

  CALL f_glrt_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE glrt_information

!  ------------------------------------
!  C interface to fortran glrt_terminate
!  ------------------------------------

  SUBROUTINE glrt_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_GLRT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( glrt_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( glrt_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_glrt_full_data_type ), pointer :: fdata
  TYPE ( f_glrt_control_type ) :: fcontrol
  TYPE ( f_glrt_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_glrt_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE glrt_terminate
