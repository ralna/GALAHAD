! THIS VERSION: GALAHAD 5.4 - 2025-11-15 AT 14:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  T R S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Hussam Al Daas, Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 5.4. November 15th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_TREK_precision_ciface
!   USE GALAHAD_KINDS_precision, ONLY: ip_, ipc_, rpc_,                        &
!       C_BOOL, C_CHAR, C_PTR, C_NULL_CHAR, C_NULL_PTR, C_LOC, C_F_POINTER
!   USE GALAHAD_common_ciface, ONLY:  strlen, opt_strlen, cstr_to_fchar
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_TREK_precision, ONLY:                                          &
        f_trek_control_type => TREK_control_type,                              &
        f_trek_time_type => TREK_time_type,                                    &
        f_trek_inform_type => TREK_inform_type,                                &
        f_trek_full_data_type => TREK_full_data_type,                          &
        f_trek_initialize => TREK_initialize,                                  &
        f_trek_read_specfile => TREK_read_specfile,                            &
        f_trek_import => TREK_import,                                          &
        f_trek_s_import => TREK_s_import,                                      &
        f_trek_solve_problem => TREK_solve_problem,                            &
        f_trek_reset_control => TREK_reset_control,                            &
        f_trek_information => TREK_information,                                &
        f_trek_terminate => TREK_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in => copy_inform_in,                                  &
        copy_sls_inform_out => copy_inform_out,                                &
        copy_sls_control_in => copy_control_in,                                &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_TRS_precision_ciface, ONLY:                                    &
        trs_inform_type,                                                       &
        trs_control_type,                                                      &
        copy_trs_inform_in => copy_inform_in,                                  &
        copy_trs_inform_out => copy_inform_out,                                &
        copy_trs_control_in => copy_control_in,                                &
        copy_trs_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: trek_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: eks_max
      INTEGER ( KIND = ipc_ ) :: it_max
      REAL ( KIND = rpc_ ) :: f
      REAL ( KIND = rpc_ ) :: reduction
      REAL ( KIND = rpc_ ) :: stop_residual
      LOGICAL ( KIND = C_BOOL ) :: reorthogonalize
      LOGICAL ( KIND = C_BOOL ) :: s_version_52
      LOGICAL ( KIND = C_BOOL ) :: perturb_c
      LOGICAL ( KIND = C_BOOL ) :: stop_check_all_orders
      LOGICAL ( KIND = C_BOOL ) :: new_radius
      LOGICAL ( KIND = C_BOOL ) :: new_values
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: linear_solver_for_s
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( sls_control_type ) :: sls_s_control
      TYPE ( trs_control_type ) :: trs_control
    END TYPE trek_control_type

    TYPE, BIND( C ) :: trek_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: assemble
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_assemble
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE trek_time_type

    TYPE, BIND( C ) :: trek_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: n_vec
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: radius
      REAL ( KIND = rpc_ ) :: next_radius
      REAL ( KIND = rpc_ ) :: error
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( trek_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( sls_inform_type ) :: sls_s_inform
      TYPE ( trs_inform_type ) :: trs_inform
    END TYPE trek_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( trek_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_trek_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%eks_max = ccontrol%eks_max
    fcontrol%it_max = ccontrol%it_max

    ! Reals
    fcontrol%f = ccontrol%f
    fcontrol%reduction = ccontrol%reduction
    fcontrol%stop_residual = ccontrol%stop_residual

    ! Logicals
    fcontrol%reorthogonalize = ccontrol%reorthogonalize
    fcontrol%s_version_52 = ccontrol%s_version_52
    fcontrol%perturb_c = ccontrol%perturb_c
    fcontrol%stop_check_all_orders = ccontrol%stop_check_all_orders
    fcontrol%new_radius = ccontrol%new_radius
    fcontrol%new_values = ccontrol%new_values
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_sls_control_in( ccontrol%sls_s_control, fcontrol%sls_s_control )
    CALL copy_trs_control_in( ccontrol%trs_control, fcontrol%trs_control )

    ! Strings
    DO i = 1, LEN( fcontrol%linear_solver )
      IF ( ccontrol%linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%linear_solver( i : i ) = ccontrol%linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%linear_solver_for_s )
      IF ( ccontrol%linear_solver_for_s( i ) == C_NULL_CHAR ) EXIT
      fcontrol%linear_solver_for_s( i : i ) = ccontrol%linear_solver_for_s( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_trek_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( trek_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%eks_max = fcontrol%eks_max
    ccontrol%it_max = fcontrol%it_max

    ! Reals
    ccontrol%f = fcontrol%f
    ccontrol%reduction = fcontrol%reduction
    ccontrol%stop_residual = fcontrol%stop_residual

    ! Logicals
    ccontrol%reorthogonalize = fcontrol%reorthogonalize
    ccontrol%s_version_52 = fcontrol%s_version_52
    ccontrol%perturb_c = fcontrol%perturb_c
    ccontrol%stop_check_all_orders = fcontrol%stop_check_all_orders
    ccontrol%new_radius = fcontrol%new_radius
    ccontrol%new_values = fcontrol%new_values
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_sls_control_out( fcontrol%sls_s_control, ccontrol%sls_s_control )
    CALL copy_trs_control_out( fcontrol%trs_control, ccontrol%trs_control )

    ! Strings
    l = LEN( fcontrol%linear_solver )
    DO i = 1, l
      ccontrol%linear_solver( i ) = fcontrol%linear_solver( i : i )
    END DO
    ccontrol%linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%linear_solver_for_s )
    DO i = 1, l
      ccontrol%linear_solver_for_s( i ) = fcontrol%linear_solver_for_s( i : i )
    END DO
    ccontrol%linear_solver_for_s( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( trek_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_trek_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%assemble = ctime%assemble
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_assemble = ctime%clock_assemble
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_trek_time_type ), INTENT( IN ) :: ftime
    TYPE ( trek_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%assemble = ftime%assemble
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_assemble = ftime%clock_assemble
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( trek_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_trek_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%n_vec = cinform%n_vec

    ! Reals
    finform%obj = cinform%obj
    finform%x_norm = cinform%x_norm
    finform%multiplier = cinform%multiplier
    finform%radius = cinform%radius
    finform%next_radius = cinform%next_radius
    finform%error = cinform%error

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_sls_inform_in( cinform%sls_s_inform, finform%sls_s_inform )
    CALL copy_trs_inform_in( cinform%trs_inform, finform%trs_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_trek_inform_type ), INTENT( IN ) :: finform
    TYPE ( trek_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%n_vec = finform%n_vec

    ! Reals
    cinform%obj = finform%obj
    cinform%x_norm = finform%x_norm
    cinform%multiplier = finform%multiplier
    cinform%radius = finform%radius
    cinform%next_radius = finform%next_radius
    cinform%error = finform%error

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_sls_inform_out( finform%sls_s_inform, cinform%sls_s_inform )
    CALL copy_trs_inform_out( finform%trs_inform, cinform%trs_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_TREK_precision_ciface

!  --------------------------------------
!  C interface to fortran trek_initialize
!  --------------------------------------

  SUBROUTINE trek_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( trek_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_trek_full_data_type ), POINTER :: fdata
  TYPE ( f_trek_control_type ) :: fcontrol
  TYPE ( f_trek_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_trek_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE trek_initialize

!  -----------------------------------------
!  C interface to fortran trek_read_specfile
!  -----------------------------------------

  SUBROUTINE trek_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( trek_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_trek_control_type ) :: fcontrol
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

  CALL f_trek_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE trek_read_specfile

!  ----------------------------------
!  C interface to fortran trek_inport
!  ----------------------------------

  SUBROUTINE trek_import( ccontrol, cdata, status, n,                          &
                          chtype, hne, hrow, hcol, hptr  ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( trek_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_trek_control_type ) :: fcontrol
  TYPE ( f_trek_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required TREK structure

  CALL f_trek_import( fcontrol, fdata, status, n,                              &
                      fhtype, hne, hrow, hcol, hptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE trek_import

!  -----------------------------------
!  C interface to fortran trek_inport_s
!  -----------------------------------

  SUBROUTINE trek_s_import( cdata, status, n, cstype, sne, srow, scol,         &
                            sptr ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, sne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( sne ), OPTIONAL :: srow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( sne ), OPTIONAL :: scol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: sptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cstype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cstype ) ) :: fstype
  TYPE ( f_trek_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  convert C string to Fortran string

  fstype = cstr_to_fchar( cstype )

!  import the problem data into the required TREK structure

  CALL f_trek_s_import( fdata, status, fstype, sne, srow, scol, sptr )
  RETURN

  END SUBROUTINE trek_s_import

!  -----------------------------------------
!  C interface to fortran trek_reset_control
!  -----------------------------------------

  SUBROUTINE trek_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( trek_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_trek_control_type ) :: fcontrol
  TYPE ( f_trek_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_trek_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE trek_reset_control

!  -----------------------------------------
!  C interface to fortran trek_solve_problem
!  -----------------------------------------

  SUBROUTINE trek_solve_problem( cdata, status, n, hne, hval, c, radius,       &
                                 x, sne, sval ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne, sne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( sne ) :: sval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: radius
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_trek_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem

  CALL f_TREK_solve_problem( fdata, status, hval, c, radius, x, sval )
  RETURN

  END SUBROUTINE trek_solve_problem

!  ---------------------------------------
!  C interface to fortran trek_information
!  ---------------------------------------

  SUBROUTINE trek_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( trek_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_trek_full_data_type ), pointer :: fdata
  TYPE ( f_trek_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain TREK solution information

  CALL f_trek_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE trek_information

!  ------------------------------------
!  C interface to fortran trek_terminate
!  ------------------------------------

  SUBROUTINE trek_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_TREK_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( trek_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( trek_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_trek_full_data_type ), pointer :: fdata
  TYPE ( f_trek_control_type ) :: fcontrol
  TYPE ( f_trek_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_trek_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE trek_terminate
