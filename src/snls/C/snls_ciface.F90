! THIS VERSION: GALAHAD 5.5 - 2026-03-08 AT 09:10 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  B N L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 5.5. February 29th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SNLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SNLS_precision, ONLY:                                          &
        f_snls_control_type               => SNLS_control_type,                &
        f_snls_time_type                  => SNLS_time_type,                   &
        f_snls_inform_type                => SNLS_inform_type,                 &
        f_snls_full_data_type             => SNLS_full_data_type,              &
        f_snls_initialize                 => SNLS_initialize,                  &
        f_snls_read_specfile              => SNLS_read_specfile,               &
        f_snls_import                     => SNLS_import,                      &
        f_snls_import_without_jac         => SNLS_import_without_jac,          &
        f_snls_reset_control              => SNLS_reset_control,               &
        f_snls_solve_with_jac             => SNLS_solve_with_jac,              &
        f_snls_solve_with_jacprod         => SNLS_solve_with_jacprod,          &
        f_snls_solve_reverse_with_jac     => SNLS_solve_reverse_with_jac,      &
        f_snls_solve_reverse_with_jacprod => SNLS_solve_reverse_with_jacprod,  &
        f_snls_information                => SNLS_information,                 &
        f_snls_terminate                  => SNLS_terminate
    USE GALAHAD_USERDATA_precision, ONLY:                                      &
        f_userdata_type => USERDATA_type

    USE GALAHAD_SLLS_precision_ciface, ONLY:                                   &
        slls_inform_type,                                                      &
        slls_control_type,                                                     &
        copy_slls_inform_in   => copy_inform_in,                               &
        copy_slls_inform_out  => copy_inform_out,                              &
        copy_slls_control_in  => copy_control_in,                              &
        copy_slls_control_out => copy_control_out

    USE GALAHAD_SLLSB_precision_ciface, ONLY:                                  &
        sllsb_inform_type,                                                     &
        sllsb_control_type,                                                    &
        copy_sllsb_inform_in   => copy_inform_in,                              &
        copy_sllsb_inform_out  => copy_inform_out,                             &
        copy_sllsb_control_in  => copy_control_in,                             &
        copy_sllsb_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: snls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: alive_unit
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      INTEGER ( KIND = ipc_ ) :: jacobian_available
      INTEGER ( KIND = ipc_ ) :: subproblem_solver
      INTEGER ( KIND = ipc_ ) :: non_monotone
      INTEGER ( KIND = ipc_ ) :: weight_update_strategy
      REAL ( KIND = rpc_ ) :: stop_r_absolute
      REAL ( KIND = rpc_ ) :: stop_r_relative
      REAL ( KIND = rpc_ ) :: stop_pg_absolute
      REAL ( KIND = rpc_ ) :: stop_pg_relative
      REAL ( KIND = rpc_ ) :: stop_s
      REAL ( KIND = rpc_ ) :: stop_pg_switch
      REAL ( KIND = rpc_ ) :: initial_weight
      REAL ( KIND = rpc_ ) :: minimum_weight
      REAL ( KIND = rpc_ ) :: eta_successful
      REAL ( KIND = rpc_ ) :: eta_very_successful
      REAL ( KIND = rpc_ ) :: eta_too_successful
      REAL ( KIND = rpc_ ) :: weight_decrease_min
      REAL ( KIND = rpc_ ) :: weight_decrease
      REAL ( KIND = rpc_ ) :: weight_increase
      REAL ( KIND = rpc_ ) :: weight_increase_max
      REAL ( KIND = rpc_ ) :: switch_to_newton
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: newton_acceleration
      LOGICAL ( KIND = C_BOOL ) :: magic_step
      LOGICAL ( KIND = C_BOOL ) :: print_obj
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( slls_control_type ) :: SLLS_control
      TYPE ( sllsb_control_type ) :: SLLSB_control
    END TYPE snls_control_type

    TYPE, BIND( C ) :: snls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: slls
      REAL ( KIND = rpc_ ) :: sllsb
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_slls
      REAL ( KIND = rpc_ ) :: clock_sllsb
    END TYPE snls_time_type

    TYPE, BIND( C ) :: snls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 13 ) :: bad_eval
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: inner_iter
      INTEGER ( KIND = ipc_ ) :: r_eval
      INTEGER ( KIND = ipc_ ) :: jr_eval
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: norm_r
      REAL ( KIND = rpc_ ) :: norm_g
      REAL ( KIND = rpc_ ) :: norm_pg
      REAL ( KIND = rpc_ ) :: weight
      TYPE ( snls_time_type ) :: time
      TYPE ( slls_inform_type ) :: SLLS_inform
      TYPE ( sllsb_inform_type ) :: SLLSB_inform
    END TYPE snls_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_R( n, m_r, x, r, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r
        REAL ( KIND = rpc_ ), DIMENSION( n ),INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( m_r ),INTENT( OUT ) :: r
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_R
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_Jr( n, m_r, jr_ne, x, jr_val,                              &
                        userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, jr_ne
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( jr_ne ),INTENT( OUT ) :: jr_val
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_Jr
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_Jr_prod( n, m_r, x, transpose, v, p, got_jr,               &
                             userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: transpose
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( MAX( n, m_r ) ), INTENT( IN ) :: v
        REAL ( KIND = rpc_ ), DIMENSION( MAX( n, m_r ) ), INTENT( OUT ) :: p
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_jr
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_Jr_prod
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_Jr_scol( n, m_r, x, index, val, row, nz, got_jr,           &
                             userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, index
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: val
        INTEGER ( KIND = ipc_ ), DIMENSION( n ), INTENT( OUT ) :: row
        INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nz
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_jr
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_Jr_scol
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_Jr_sprod( n, m_r, x, transpose, v, p, free, n_free,        &
                              got_jr, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, n_free
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: transpose
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( MAX( n, m_r ) ), INTENT( IN ) :: v
        REAL ( KIND = rpc_ ), DIMENSION( MAX( n, m_r ) ), INTENT( OUT ) :: p
        INTEGER ( KIND = ipc_ ), DIMENSION( n_free ), INTENT( IN ) :: free
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ) :: got_jr
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_Jr_sprod
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( snls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_snls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%alive_unit = ccontrol%alive_unit
    fcontrol%jacobian_available = ccontrol%jacobian_available
    fcontrol%subproblem_solver = ccontrol%subproblem_solver
    fcontrol%non_monotone = ccontrol%non_monotone
    fcontrol%weight_update_strategy = ccontrol%weight_update_strategy

    ! Reals
    fcontrol%stop_r_absolute = ccontrol%stop_r_absolute
    fcontrol%stop_pg_absolute = ccontrol%stop_pg_absolute
    fcontrol%stop_r_relative = ccontrol%stop_r_relative
    fcontrol%stop_pg_relative = ccontrol%stop_pg_relative
    fcontrol%stop_s = ccontrol%stop_s
    fcontrol%stop_pg_switch = ccontrol%stop_pg_switch
    fcontrol%initial_weight = ccontrol%initial_weight
    fcontrol%minimum_weight = ccontrol%minimum_weight
    fcontrol%eta_successful = ccontrol%eta_successful
    fcontrol%eta_very_successful = ccontrol%eta_very_successful
    fcontrol%eta_too_successful = ccontrol%eta_too_successful
    fcontrol%weight_decrease_min = ccontrol%weight_decrease_min
    fcontrol%weight_decrease = ccontrol%weight_decrease
    fcontrol%weight_increase = ccontrol%weight_increase
    fcontrol%weight_increase_max = ccontrol%weight_increase_max
    fcontrol%switch_to_newton = ccontrol%switch_to_newton
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%magic_step = ccontrol%magic_step
    fcontrol%print_obj = ccontrol%print_obj
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_slls_control_in( ccontrol%slls_control, fcontrol%slls_control )
    CALL copy_sllsb_control_in( ccontrol%sllsb_control, fcontrol%sllsb_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_snls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( snls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing

    IF ( PRESENT( f_indexing ) )  ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%alive_unit = fcontrol%alive_unit
    ccontrol%jacobian_available = fcontrol%jacobian_available
    ccontrol%subproblem_solver = fcontrol%subproblem_solver
    ccontrol%non_monotone = fcontrol%non_monotone
    ccontrol%weight_update_strategy = fcontrol%weight_update_strategy

    ! Reals
    ccontrol%stop_r_absolute = fcontrol%stop_r_absolute
    ccontrol%stop_pg_absolute = fcontrol%stop_pg_absolute
    ccontrol%stop_r_relative = fcontrol%stop_r_relative
    ccontrol%stop_pg_relative = fcontrol%stop_pg_relative
    ccontrol%stop_s = fcontrol%stop_s
    ccontrol%stop_pg_switch = fcontrol%stop_pg_switch
    ccontrol%initial_weight = fcontrol%initial_weight
    ccontrol%minimum_weight = fcontrol%minimum_weight
    ccontrol%eta_successful = fcontrol%eta_successful
    ccontrol%eta_very_successful = fcontrol%eta_very_successful
    ccontrol%eta_too_successful = fcontrol%eta_too_successful
    ccontrol%weight_decrease_min = fcontrol%weight_decrease_min
    ccontrol%weight_decrease = fcontrol%weight_decrease
    ccontrol%weight_increase = fcontrol%weight_increase
    ccontrol%weight_increase_max = fcontrol%weight_increase_max
    ccontrol%switch_to_newton = fcontrol%switch_to_newton
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%magic_step = fcontrol%magic_step
    ccontrol%print_obj = fcontrol%print_obj
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_slls_control_out( fcontrol%slls_control,                         &
                                ccontrol%slls_control )
    CALL copy_sllsb_control_out( fcontrol%sllsb_control,                       &
                               ccontrol%sllsb_control )

    ! Strings
    l = LEN( fcontrol%alive_file )
    DO i = 1, l
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( snls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_snls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%slls = ctime%slls
    ftime%sllsb = ctime%sllsb
    ftime%clock_total = ctime%clock_total
    ftime%clock_slls = ctime%clock_slls
    ftime%clock_sllsb = ctime%clock_sllsb
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_snls_time_type ), INTENT( IN ) :: ftime
    TYPE ( snls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%slls = ftime%slls
    ctime%sllsb = ftime%sllsb
    ctime%clock_total = ftime%clock_total
    ctime%clock_slls = ftime%clock_slls
    ctime%clock_sllsb = ftime%clock_sllsb
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( snls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_snls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%inner_iter = cinform%inner_iter
    finform%r_eval = cinform%r_eval
    finform%jr_eval = cinform%jr_eval

    ! Reals
    finform%obj = cinform%obj
    finform%norm_r = cinform%norm_r
    finform%norm_g = cinform%norm_g
    finform%norm_pg = cinform%norm_pg
    finform%weight = cinform%weight

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_slls_inform_in( cinform%slls_inform, finform%slls_inform )
    CALL copy_sllsb_inform_in( cinform%sllsb_inform, finform%sllsb_inform )

    ! Strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, 13
      IF ( cinform%bad_eval( i ) == C_NULL_CHAR ) EXIT
      finform%bad_eval( i : i ) = cinform%bad_eval( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_snls_inform_type ), INTENT( IN ) :: finform
    TYPE ( snls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%inner_iter = finform%inner_iter
    cinform%r_eval = finform%r_eval
    cinform%jr_eval = finform%jr_eval

    ! Reals
    cinform%obj = finform%obj
    cinform%norm_r = finform%norm_r
    cinform%norm_g = finform%norm_g
    cinform%norm_pg = finform%norm_pg
    cinform%weight = finform%weight

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_slls_inform_out( finform%slls_inform, cinform%slls_inform )
    CALL copy_sllsb_inform_out( finform%sllsb_inform, cinform%sllsb_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( finform%bad_eval )
      cinform%bad_eval( i ) = finform%bad_eval( i : i )
    END DO
    cinform%bad_eval( LEN( finform%bad_eval ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SNLS_precision_ciface

!  -------------------------------------
!  C interface to fortran snls_initialize
!  -------------------------------------

  SUBROUTINE snls_initialize( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( snls_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( snls_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  TYPE ( f_snls_control_type ) :: fcontrol
  TYPE ( f_snls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_snls_initialize( fdata, fcontrol, finform )

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE snls_initialize

!  ----------------------------------------
!  C interface to fortran snls_read_specfile
!  ----------------------------------------

  SUBROUTINE snls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( snls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_snls_control_type ) :: fcontrol
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

  CALL f_snls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE snls_read_specfile

!  ----------------------------------
!  C interface to fortran snls_inport
! - ---------------------------------


  SUBROUTINE snls_import( ccontrol, cdata, status, n, m_r, m_c, cjr_type,      &
                          jr_ne, jr_row, jr_col, jr_ptr_ne, jr_ptr,            &
                          cohort ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( snls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c, jr_ne, jr_ptr_ne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( jr_ne ), OPTIONAL :: jr_row
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( jr_ne ), OPTIONAL :: jr_col
  INTEGER ( KIND = ipc_ ), INTENT( IN ),                                       &
                           DIMENSION( jr_ptr_ne ), OPTIONAL :: jr_ptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cjr_type
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: cohort

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cjr_type ) ) :: fjr_type
  TYPE ( f_snls_control_type ) :: fcontrol
  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fjr_type = cstr_to_fchar( cjr_type )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

!  import the problem data into the required SNLS structure

  CALL f_snls_import( fcontrol, fdata, status, n, m_r, m_c,                    &
                      fjr_type, jr_ne, jr_row, jr_col, jr_ptr, cohort )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE snls_import

!  ----------------------------------------------
!  C interface to fortran snls_inport_without_jac
!  ----------------------------------------------

  SUBROUTINE snls_import_without_jac( ccontrol, cdata, status, n, m_r, m_c,    &
                                      cohort ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( snls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: cohort

!  local variables

  TYPE ( f_snls_control_type ) :: fcontrol
  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

  CALL f_snls_import_without_jac( fcontrol, fdata, status, n, m_r, m_c,        &
                                  COHORT = cohort )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE snls_import_without_jac

!  ----------------------------------------
!  C interface to fortran snls_reset_control
!  ----------------------------------------

  SUBROUTINE snls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( snls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_snls_control_type ) :: fcontrol
  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_snls_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE snls_reset_control

!  ------------------------------------------
!  C interface to fortran snls_solve_with_jac
!  ------------------------------------------

  SUBROUTINE snls_solve_with_jac( cdata, cuserdata, status, n, m_r, m_c,       &
                                  x, y, z, r, g, x_stat, ceval_r,              &
                                  jr_ne, ceval_jr, w ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c, jr_ne
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_c ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_r ) :: r
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: x_stat
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( m_r ) :: w
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_r, ceval_jr

!  local variables

  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_r ), POINTER :: feval_r
  PROCEDURE( eval_jr ), POINTER :: feval_jr

!  ignore Fortran userdata type (not interoperable)

  TYPE ( f_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_r, feval_r )
  CALL C_F_PROCPOINTER( ceval_jr, feval_jr )

!  solve the problem when the Hessian is explicitly available

  CALL f_snls_solve_with_jac( fdata, fuserdata, status, x, y, z, r, g, x_stat, &
                              wrap_eval_r, wrap_eval_jr, W = w )

  RETURN

!  wrappers

  CONTAINS

!  eval_c wrapper

    SUBROUTINE wrap_eval_r( status, x, userdata, r )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: r

!  call C interoperable eval_c

!   write(6, "( ' X in wrap_eval_c = ', 2ES12.4 )" ) x
    status = feval_r( n, m_r, x, r, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_r

!  eval_j wrapper

    SUBROUTINE wrap_eval_jr( status, x, userdata, jr_val )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: jr_val

!  Call C interoperable eval_jr
    status = feval_jr( n, m_r, jr_ne, x, jr_val, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_jr

  END SUBROUTINE snls_solve_with_jac

!  ----------------------------------------------
!  C interface to fortran snls_solve_with_jacprod
!  ----------------------------------------------

  SUBROUTINE snls_solve_with_jacprod( cdata, cuserdata, status, n, m_r, m_c,   &
                                      x, y, z, r, g, x_stat, ceval_r,          &
                                      ceval_jr_prod, ceval_jr_scol,            &
                                      ceval_jr_sprod, w ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_c ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_r ) :: r
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: x_stat
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( m_r ) :: w
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_r, ceval_jr_prod
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_jr_scol, ceval_jr_sprod

!  local variables

  TYPE ( f_snls_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_r ), POINTER :: feval_r
  PROCEDURE( eval_jr_prod ), POINTER :: feval_jr_prod
  PROCEDURE( eval_jr_scol ), POINTER :: feval_jr_scol
  PROCEDURE( eval_jr_sprod ), POINTER :: feval_jr_sprod
  LOGICAL :: f_indexing

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

  f_indexing = fdata%f_indexing

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_r, feval_r )
  CALL C_F_PROCPOINTER( ceval_jr_prod, feval_jr_prod )
  CALL C_F_PROCPOINTER( ceval_jr_scol, feval_jr_scol )
  CALL C_F_PROCPOINTER( ceval_jr_sprod, feval_jr_sprod )

!  solve the problem when the Hessian is only available via products

  CALL f_snls_solve_with_jacprod( fdata, fuserdata, status, x, y, z, r, g,     &
                                  x_stat, wrap_eval_r, wrap_eval_jr_prod,      &
                                  wrap_eval_jr_scol, wrap_eval_jr_sprod,       &
                                  W = w )
  RETURN

!  wrappers

  CONTAINS

!  eval_c wrapper

    SUBROUTINE wrap_eval_r( status, x, userdata, r )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: r

!  call C interoperable eval_r

    status = feval_r( n, m_r, x, r, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_r

!  eval_jprod wrapper

    SUBROUTINE wrap_eval_jr_prod( status, x, userdata, ftranspose, v, p,       &
                                  fgot_jr )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: p
    LOGICAL, INTENT( IN ) :: ftranspose
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_jr
    LOGICAL ( KIND = C_BOOL ) :: cgot_jr, ctranspose

!  call C interoperable eval_jr_prod

    ctranspose = ftranspose
    IF ( PRESENT( fgot_jr ) ) THEN
      cgot_jr = fgot_jr
    ELSE
      cgot_jr = .FALSE.
    END IF
    status = feval_jr_prod( n, m_r, x, ctranspose, v, p, cgot_jr, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_jr_prod

    SUBROUTINE wrap_eval_Jr_scol( status, x, userdata, index, val, row, nz,    &
                                  fgot_jr )
    USE GALAHAD_USERDATA_precision
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    INTEGER ( KIND = ipc_ ), INTENT( IN ) :: index
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: val
    INTEGER ( KIND = ipc_ ), DIMENSION( : ), INTENT( INOUT ) :: row
    INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: nz
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_jr
    LOGICAL ( KIND = C_BOOL ) :: cgot_jr

!  call C interoperable eval_jr_scol

    IF ( PRESENT( fgot_jr ) ) THEN
      cgot_jr = fgot_jr
    ELSE
      cgot_jr = .FALSE.
    END IF

    IF ( f_indexing ) THEN
      status = feval_Jr_scol( n, m_r, x, index, val, row, nz, cgot_jr,         &
                              cuserdata )
    ELSE
      status = feval_Jr_scol( n, m_r, x, index - 1, val, row, nz, cgot_jr,     &
                              cuserdata )
      row( : nz ) = row( : nz ) + 1
    END IF
    RETURN

    END SUBROUTINE wrap_eval_Jr_scol

    SUBROUTINE wrap_eval_Jr_sprod( status, x, userdata, ftranspose, v, p,      &
                                   free, n_free, fgot_jr )
    USE GALAHAD_USERDATA_precision
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_userdata_type ), INTENT( INOUT ) :: userdata
    LOGICAL, INTENT( IN ) :: ftranspose
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: p
    INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( : ) :: free
    INTEGER ( KIND = ipc_ ), INTENT( IN ) :: n_free
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_jr
    LOGICAL ( KIND = C_BOOL ) :: ctranspose, cgot_jr

!  call C interoperable eval_jr_sprod

    ctranspose = ftranspose
    IF ( PRESENT( fgot_jr ) ) THEN
      cgot_jr = fgot_jr
    ELSE
      cgot_jr = .FALSE.
    END IF

    IF ( f_indexing ) THEN
      status = feval_Jr_sprod( n, m_r, x, ctranspose, v, p, free, n_free,      &
                               cgot_jr, cuserdata )
    ELSE
      status = feval_Jr_sprod( n, m_r, x, ctranspose, v, p, free - 1, n_free,  &
                               cgot_jr, cuserdata )
    END IF
    RETURN

    END SUBROUTINE wrap_eval_Jr_sprod

  END SUBROUTINE snls_solve_with_jacprod

!  --------------------------------------------------
!  C interface to fortran snls_solve_reverse_with_jac
!  --------------------------------------------------

  SUBROUTINE snls_solve_reverse_with_jac( cdata, status, eval_status,          &
                                          n, m_r, m_c, x, y, z, r, g, x_stat,  &
                                          jr_ne, jr_val, w ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c, jr_ne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_c ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m_r ) :: r
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: x_stat
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( m_r ) :: w
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( jr_ne ) :: jr_val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_snls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when the Hessian is available by reverse communication

  CALL f_snls_solve_reverse_with_jac( fdata, status, eval_status,              &
                                      x, y, z, r, g, x_stat, jr_val, W = w )
  RETURN

  END SUBROUTINE snls_solve_reverse_with_jac

!  ------------------------------------------------------
!  C interface to fortran snls_solve_reverse_with_jacprod
!  ------------------------------------------------------

  SUBROUTINE snls_solve_reverse_with_jacprod( cdata, status, eval_status,      &
                                              n, m_r, m_c, x, y, z, r, g,      &
                                              x_stat, v, iv, lvl, lvu, index,  &
                                              p, ip, lp, w ) BIND( C )
                                             
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m_r, m_c
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: lvl, lvu, index
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: lp
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m_c ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m_r ) :: r
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: x_stat
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( MAX( n, m_r ) ) :: ip
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( MAX( n, m_r ) ) :: iv
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( MAX( n, m_r ) ) :: p
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( MAX( n, m_r ) ) :: v
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( m_r ) :: w
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_snls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when Hessian products are available by reverse
!  communication

  CALL f_snls_solve_reverse_with_jacprod( fdata, status, eval_status,          &
                                          x, y, z, r, g, x_stat,               &
                                          v, iv, lvl, lvu, index,              &
                                          p, ip, lp, W = w )
  RETURN

  END SUBROUTINE snls_solve_reverse_with_jacprod

!  --------------------------------------
!  C interface to fortran snls_information
!  --------------------------------------

  SUBROUTINE snls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( snls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_snls_full_data_type ), pointer :: fdata
  TYPE ( f_snls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain SNLS solution information

  CALL f_snls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

  RETURN

  END SUBROUTINE snls_information

!  ------------------------------------
!  C interface to fortran snls_terminate
!  ------------------------------------

  SUBROUTINE snls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SNLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( snls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( snls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_snls_full_data_type ), pointer :: fdata
  TYPE ( f_snls_control_type ) :: fcontrol
  TYPE ( f_snls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_snls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE snls_terminate
