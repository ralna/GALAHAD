! THIS VERSION: GALAHAD 5.3 - 2025-07-25 AT 08:20 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  E X P O    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. August 22nd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_EXPO_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_EXPO_precision, ONLY:                                          &
        f_expo_control_type              => EXPO_control_type,                 &
        f_expo_time_type                 => EXPO_time_type,                    &
        f_expo_inform_type               => EXPO_inform_type,                  &
        f_expo_full_data_type            => EXPO_full_data_type,               &
        f_expo_initialize                => EXPO_initialize,                   &
        f_expo_read_specfile             => EXPO_read_specfile,                &
        f_expo_import                    => EXPO_import,                       &
        f_expo_reset_control             => EXPO_reset_control,                &
        f_expo_solve_hessian_direct      => EXPO_solve_hessian_direct,         &
        f_expo_information               => EXPO_information,                  &
        f_expo_terminate                 => EXPO_terminate
    USE GALAHAD_USERDATA_precision, ONLY:                                      &
        f_galahad_userdata_type => GALAHAD_userdata_type

    USE GALAHAD_BSC_precision_ciface, ONLY:                                    &
        bsc_inform_type,                                                       &
        bsc_control_type,                                                      &
        copy_bsc_inform_in   => copy_inform_in,                                &
        copy_bsc_inform_out  => copy_inform_out,                               &
        copy_bsc_control_in  => copy_control_in,                               &
        copy_bsc_control_out => copy_control_out

    USE GALAHAD_TRU_precision_ciface, ONLY:                                    &
        tru_inform_type,                                                       &
        tru_control_type,                                                      &
        copy_tru_inform_in   => copy_inform_in,                                &
        copy_tru_inform_out  => copy_inform_out,                               &
        copy_tru_control_in  => copy_control_in,                               &
        copy_tru_control_out => copy_control_out

    USE GALAHAD_SSLS_precision_ciface, ONLY:                                   &
        ssls_inform_type,                                                      &
        ssls_control_type,                                                     &
        copy_ssls_inform_in   => copy_inform_in,                               &
        copy_ssls_inform_out  => copy_inform_out,                              &
        copy_ssls_control_in  => copy_control_in,                              &
        copy_ssls_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: expo_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: max_it
      INTEGER ( KIND = ipc_ ) :: max_eval
      INTEGER ( KIND = ipc_ ) :: alive_unit
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      INTEGER ( KIND = ipc_ ) :: update_multipliers_itmin
      REAL ( KIND = rpc_ ) :: update_multipliers_tol
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: stop_abs_p
      REAL ( KIND = rpc_ ) :: stop_rel_p
      REAL ( KIND = rpc_ ) :: stop_abs_d
      REAL ( KIND = rpc_ ) :: stop_rel_d
      REAL ( KIND = rpc_ ) :: stop_abs_c
      REAL ( KIND = rpc_ ) :: stop_rel_c
      REAL ( KIND = rpc_ ) :: stop_s
      REAL ( KIND = rpc_ ) :: initial_mu
      REAL ( KIND = rpc_ ) :: mu_reduce
      REAL ( KIND = rpc_ ) :: obj_unbounded
      REAL ( KIND = rpc_ ) :: try_advanced_start
      REAL ( KIND = rpc_ ) :: try_sqp_start
      REAL ( KIND = rpc_ ) :: stop_advance_start
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: hessian_available
      LOGICAL ( KIND = C_BOOL ) :: subproblem_direct
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( bsc_control_type ) :: bsc_control
      TYPE ( tru_control_type ) :: tru_control
      TYPE ( ssls_control_type ) :: ssls_control
    END TYPE expo_control_type

    TYPE, BIND( C ) :: expo_time_type
      REAL ( KIND = spc_ ) :: total
      REAL ( KIND = spc_ ) :: preprocess
      REAL ( KIND = spc_ ) :: analyse
      REAL ( KIND = spc_ ) :: factorize
      REAL ( KIND = spc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_preprocess
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE expo_time_type

    TYPE, BIND( C ) :: expo_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 13 ) :: bad_eval
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: fc_eval
      INTEGER ( KIND = ipc_ ) :: gj_eval
      INTEGER ( KIND = ipc_ ) :: hl_eval
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: primal_infeasibility
      REAL ( KIND = rpc_ ) :: dual_infeasibility
      REAL ( KIND = rpc_ ) :: complementary_slackness
      TYPE ( expo_time_type ) :: time
      TYPE ( bsc_inform_type ) :: bsc_inform
      TYPE ( tru_inform_type ) :: tru_inform
      TYPE ( ssls_inform_type ) :: ssls_inform
    END TYPE expo_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_FC( n, m, x, f, c, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m
        REAL ( KIND = rpc_ ), DIMENSION( n ),INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
        REAL ( KIND = rpc_ ), DIMENSION( m ),INTENT( OUT ) :: c
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_FC
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_GJ( n, m, jne, x, g, jval, userdata ) RESULT( status )     &
          BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, jne
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: g
        REAL ( KIND = rpc_ ), DIMENSION( jne ),INTENT( OUT ) :: jval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_GJ
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_HL( n, m, hne, x, y, hval,                                 &
                        userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( m ), INTENT( IN ) :: y
        REAL ( KIND = rpc_ ), DIMENSION( hne ), INTENT( OUT ) :: hval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_HL
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( expo_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_expo_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%max_it = ccontrol%max_it
    fcontrol%max_eval = ccontrol%max_eval
    fcontrol%alive_unit = ccontrol%alive_unit
    fcontrol%update_multipliers_itmin = ccontrol%update_multipliers_itmin

    ! Reals
    fcontrol%update_multipliers_tol = ccontrol%update_multipliers_tol
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_abs_p = ccontrol%stop_abs_p
    fcontrol%stop_rel_p = ccontrol%stop_rel_p
    fcontrol%stop_abs_d = ccontrol%stop_abs_d
    fcontrol%stop_rel_d = ccontrol%stop_rel_d
    fcontrol%stop_abs_c = ccontrol%stop_abs_c
    fcontrol%stop_rel_c = ccontrol%stop_rel_c
    fcontrol%stop_s = ccontrol%stop_s
    fcontrol%initial_mu = ccontrol%initial_mu
    fcontrol%mu_reduce = ccontrol%mu_reduce
    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%try_advanced_start = ccontrol%try_advanced_start
    fcontrol%try_sqp_start = ccontrol%try_sqp_start
    fcontrol%stop_advance_start = ccontrol%stop_advance_start
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%hessian_available = ccontrol%hessian_available
    fcontrol%subproblem_direct = ccontrol%subproblem_direct
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_bsc_control_in( ccontrol%bsc_control, fcontrol%bsc_control )
    CALL copy_tru_control_in( ccontrol%tru_control, fcontrol%tru_control )
    CALL copy_ssls_control_in( ccontrol%ssls_control, fcontrol%ssls_control )

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_expo_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( expo_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing

    IF ( PRESENT( f_indexing ) )  ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%max_it = fcontrol%max_it
    ccontrol%max_eval = fcontrol%max_eval
    ccontrol%alive_unit = fcontrol%alive_unit
    ccontrol%update_multipliers_itmin = fcontrol%update_multipliers_itmin

    ! Reals
    ccontrol%update_multipliers_tol = fcontrol%update_multipliers_tol
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_abs_p = fcontrol%stop_abs_p
    ccontrol%stop_rel_p = fcontrol%stop_rel_p
    ccontrol%stop_abs_d = fcontrol%stop_abs_d
    ccontrol%stop_rel_d = fcontrol%stop_rel_d
    ccontrol%stop_abs_c = fcontrol%stop_abs_c
    ccontrol%stop_rel_c = fcontrol%stop_rel_c
    ccontrol%stop_s = fcontrol%stop_s
    ccontrol%initial_mu = fcontrol%initial_mu
    ccontrol%mu_reduce = fcontrol%mu_reduce
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%try_advanced_start = fcontrol%try_advanced_start
    ccontrol%try_sqp_start = fcontrol%try_sqp_start
    ccontrol%stop_advance_start = fcontrol%stop_advance_start
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%hessian_available = fcontrol%hessian_available
    ccontrol%subproblem_direct = fcontrol%subproblem_direct
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_bsc_control_out( fcontrol%bsc_control, ccontrol%bsc_control )
    CALL copy_tru_control_out( fcontrol%tru_control, ccontrol%tru_control )
    CALL copy_ssls_control_out( fcontrol%ssls_control, ccontrol%ssls_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( LEN( fcontrol%alive_file ) + 1 ) = C_NULL_CHAR
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( expo_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_expo_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_expo_time_type ), INTENT( IN ) :: ftime
    TYPE ( expo_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( expo_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_expo_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%fc_eval = cinform%fc_eval
    finform%gj_eval = cinform%gj_eval
    finform%hl_eval = cinform%hl_eval

    ! Reals
    finform%obj = cinform%obj
    finform%primal_infeasibility = cinform%primal_infeasibility
    finform%dual_infeasibility = cinform%dual_infeasibility
    finform%complementary_slackness = cinform%complementary_slackness

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_bsc_inform_in( cinform%bsc_inform, finform%bsc_inform )
    CALL copy_tru_inform_in( cinform%tru_inform, finform%tru_inform )
    CALL copy_ssls_inform_in( cinform%ssls_inform, finform%ssls_inform )

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
    TYPE ( f_expo_inform_type ), INTENT( IN ) :: finform
    TYPE ( expo_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%fc_eval = finform%fc_eval
    cinform%gj_eval = finform%gj_eval
    cinform%hl_eval = finform%hl_eval

    ! Reals
    cinform%obj = finform%obj
    cinform%primal_infeasibility = finform%primal_infeasibility
    cinform%dual_infeasibility = finform%dual_infeasibility
    cinform%complementary_slackness = finform%complementary_slackness

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_bsc_inform_out( finform%bsc_inform, cinform%bsc_inform )
    CALL copy_tru_inform_out( finform%tru_inform, cinform%tru_inform )
    CALL copy_ssls_inform_out( finform%ssls_inform, cinform%ssls_inform )

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

  END MODULE GALAHAD_EXPO_precision_ciface

!  -------------------------------------
!  C interface to fortran expo_initialize
!  -------------------------------------

  SUBROUTINE expo_initialize( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( expo_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( expo_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_expo_full_data_type ), POINTER :: fdata
  TYPE ( f_expo_control_type ) :: fcontrol
  TYPE ( f_expo_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_expo_initialize( fdata, fcontrol, finform )

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE expo_initialize

!  ----------------------------------------
!  C interface to fortran expo_read_specfile
!  ----------------------------------------

  SUBROUTINE expo_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( expo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_expo_control_type ) :: fcontrol
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

  CALL f_expo_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE expo_read_specfile

!  ---------------------------------
!  C interface to fortran expo_inport
!  ---------------------------------

  SUBROUTINE expo_import( ccontrol, cdata, status, n, m,                       &
                         cjtype, jne, jrow, jcol, jptr,                        &
                         chtype, hne, hrow, hcol, hptr ) BIND( C )
                         
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( expo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, jne, hne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: jrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( jne ), OPTIONAL :: jcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: jptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cjtype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cjtype ) ) :: fjtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_expo_control_type ) :: fcontrol
  TYPE ( f_expo_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fjtype = cstr_to_fchar( cjtype )
  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

!  import the problem data into the required EXPO structure

  CALL f_expo_import( fcontrol, fdata, status, n, m,                           &
                      fjtype, jne, jrow, jcol, jptr,                           &
                      fhtype, hne, hrow, hcol, hptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE expo_import

!  ----------------------------------------
!  C interface to fortran expo_reset_control
!  ----------------------------------------

  SUBROUTINE expo_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( expo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_expo_control_type ) :: fcontrol
  TYPE ( f_expo_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_expo_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE expo_reset_control

!  ------------------------------------------------
!  C interface to fortran expo_solve_hessian_direct
!  ------------------------------------------------

  SUBROUTINE expo_solve_hessian_direct( cdata, cuserdata, status,              &
                                        n, m, jne, hne,                        &
                                        cl, cu, xl, xu, x, y, z, c, gl,        &
                                        ceval_fc, ceval_gj, ceval_hl )         &
                                        BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, jne, hne
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: y, c
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z, gl
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_fc, ceval_gj, ceval_hl

!  local variables

  TYPE ( f_expo_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_fc ), POINTER :: feval_fc
  PROCEDURE( eval_gj ), POINTER :: feval_gj
  PROCEDURE( eval_hl ), POINTER :: feval_hl

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_fc, feval_fc )
  CALL C_F_PROCPOINTER( ceval_gj, feval_gj )
  IF ( C_ASSOCIATED( ceval_hl ) ) THEN
    CALL C_F_PROCPOINTER( ceval_hl, feval_hl )
  ELSE
    NULLIFY( feval_hl )
  END IF

!  solve the problem when the Hessian is explicitly available

  CALL f_expo_solve_hessian_direct( fdata, fuserdata, status,                  &
                                    cl, cu, xl, xu, x, y, z, c, gl,            &
                                    wrap_eval_fc, wrap_eval_gj, wrap_eval_hl )
  RETURN

!  wrappers

  CONTAINS

!  eval_c wrapper

    SUBROUTINE wrap_eval_fc( status, x, userdata, f, c )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: c

!   write(6, "( ' X in wrap_eval_fc = ', 2ES12.4 )" ) x

!  call C interoperable eval_fc
    status = feval_fc( n, m, x, f, c, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_fc

!  eval_j wrapper

    SUBROUTINE wrap_eval_gj( status, x, userdata, g, jval )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: g
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: jval

!  Call C interoperable eval_gj
    status = feval_gj( n, m, jne, x, g, jval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_gj

!  eval_H wrapper

    SUBROUTINE wrap_eval_hl( status, x, y, userdata, hval )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x, y
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: hval

!  Call C interoperable eval_h
    status = feval_hl( n, m, hne, x, y, hval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hl

  END SUBROUTINE expo_solve_hessian_direct

!  --------------------------------------
!  C interface to fortran expo_information
!  --------------------------------------

  SUBROUTINE expo_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( expo_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_expo_full_data_type ), pointer :: fdata
  TYPE ( f_expo_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain EXPO solution information

  CALL f_expo_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE expo_information

!  ------------------------------------
!  C interface to fortran expo_terminate
!  ------------------------------------

  SUBROUTINE expo_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_EXPO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( expo_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( expo_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_expo_full_data_type ), pointer :: fdata
  TYPE ( f_expo_control_type ) :: fcontrol
  TYPE ( f_expo_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_expo_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE expo_terminate
