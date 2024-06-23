! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  Q P A    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 7th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_QPA_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_QPA_precision, ONLY:                                           &
        f_qpa_control_type   => QPA_control_type,                              &
        f_qpa_time_type      => QPA_time_type,                                 &
        f_qpa_inform_type    => QPA_inform_type,                               &
        f_qpa_full_data_type => QPA_full_data_type,                            &
        f_qpa_initialize     => QPA_initialize,                                &
        f_qpa_read_specfile  => QPA_read_specfile,                             &
        f_qpa_import         => QPA_import,                                    &
        f_qpa_solve_qp       => QPA_solve_qp,                                  &
        f_qpa_solve_l1qp     => QPA_solve_l1qp,                                &
        f_qpa_solve_bcl1qp   => QPA_solve_bcl1qp,                              &
        f_qpa_reset_control  => QPA_reset_control,                             &
        f_qpa_information    => QPA_information,                               &
        f_qpa_terminate      => QPA_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: qpa_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: factor
      INTEGER ( KIND = ipc_ ) :: max_col
      INTEGER ( KIND = ipc_ ) :: max_sc
      INTEGER ( KIND = ipc_ ) :: indmin
      INTEGER ( KIND = ipc_ ) :: valmin
      INTEGER ( KIND = ipc_ ) :: itref_max
      INTEGER ( KIND = ipc_ ) :: infeas_check_interval
      INTEGER ( KIND = ipc_ ) :: cg_maxit
      INTEGER ( KIND = ipc_ ) :: precon
      INTEGER ( KIND = ipc_ ) :: nsemib
      INTEGER ( KIND = ipc_ ) :: full_max_fill
      INTEGER ( KIND = ipc_ ) :: deletion_strategy
      INTEGER ( KIND = ipc_ ) :: restore_problem
      INTEGER ( KIND = ipc_ ) :: monitor_residuals
      INTEGER ( KIND = ipc_ ) :: cold_start
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: feas_tol
      REAL ( KIND = rpc_ ) :: obj_unbounded
      REAL ( KIND = rpc_ ) :: increase_rho_g_factor
      REAL ( KIND = rpc_ ) :: infeas_g_improved_by_factor
      REAL ( KIND = rpc_ ) :: increase_rho_b_factor
      REAL ( KIND = rpc_ ) :: infeas_b_improved_by_factor
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: pivot_tol_for_dependencies
      REAL ( KIND = rpc_ ) :: zero_pivot
      REAL ( KIND = rpc_ ) :: inner_stop_relative
      REAL ( KIND = rpc_ ) :: inner_stop_absolute
      REAL ( KIND = rpc_ ) :: multiplier_tol
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: treat_zero_bounds_as_general
      LOGICAL ( KIND = C_BOOL ) :: solve_qp
      LOGICAL ( KIND = C_BOOL ) :: solve_within_bounds
      LOGICAL ( KIND = C_BOOL ) :: randomize
      LOGICAL ( KIND = C_BOOL ) :: array_syntax_worse_than_do_loop
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      LOGICAL ( KIND = C_BOOL ) :: each_interval
      TYPE ( sls_control_type ) :: sls_control
    END TYPE qpa_control_type

    TYPE, BIND( C ) :: qpa_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: preprocess
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_preprocess
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE qpa_time_type

    TYPE, BIND( C ) :: qpa_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: major_iter
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: cg_iter
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      INTEGER ( KIND = ipc_ ) :: nfacts
      INTEGER ( KIND = ipc_ ) :: nmods
      INTEGER ( KIND = ipc_ ) :: num_g_infeas
      INTEGER ( KIND = ipc_ ) :: num_b_infeas
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: infeas_g
      REAL ( KIND = rpc_ ) :: infeas_b
      REAL ( KIND = rpc_ ) :: merit
      TYPE ( qpa_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
    END TYPE qpa_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( qpa_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_qpa_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%maxit = ccontrol%maxit
    fcontrol%factor = ccontrol%factor
    fcontrol%max_col = ccontrol%max_col
    fcontrol%max_sc = ccontrol%max_sc
    fcontrol%indmin = ccontrol%indmin
    fcontrol%valmin = ccontrol%valmin
    fcontrol%itref_max = ccontrol%itref_max
    fcontrol%infeas_check_interval = ccontrol%infeas_check_interval
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%precon = ccontrol%precon
    fcontrol%nsemib = ccontrol%nsemib
    fcontrol%full_max_fill = ccontrol%full_max_fill
    fcontrol%deletion_strategy = ccontrol%deletion_strategy
    fcontrol%restore_problem = ccontrol%restore_problem
    fcontrol%monitor_residuals = ccontrol%monitor_residuals
    fcontrol%cold_start = ccontrol%cold_start
    fcontrol%sif_file_device = ccontrol%sif_file_device

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%feas_tol = ccontrol%feas_tol
    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%increase_rho_g_factor = ccontrol%increase_rho_g_factor
    fcontrol%infeas_g_improved_by_factor = ccontrol%infeas_g_improved_by_factor
    fcontrol%increase_rho_b_factor = ccontrol%increase_rho_b_factor
    fcontrol%infeas_b_improved_by_factor = ccontrol%infeas_b_improved_by_factor
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%pivot_tol_for_dependencies = ccontrol%pivot_tol_for_dependencies
    fcontrol%zero_pivot = ccontrol%zero_pivot
    fcontrol%inner_stop_relative = ccontrol%inner_stop_relative
    fcontrol%inner_stop_absolute = ccontrol%inner_stop_absolute
    fcontrol%multiplier_tol = ccontrol%multiplier_tol
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%treat_zero_bounds_as_general                                      &
      = ccontrol%treat_zero_bounds_as_general
    fcontrol%solve_qp = ccontrol%solve_qp
    fcontrol%solve_within_bounds = ccontrol%solve_within_bounds
    fcontrol%randomize = ccontrol%randomize
    fcontrol%array_syntax_worse_than_do_loop                                   &
      = ccontrol%array_syntax_worse_than_do_loop
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file
    fcontrol%each_interval = ccontrol%each_interval

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )

    ! Strings
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i )                                &
        = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_qpa_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( qpa_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%maxit = fcontrol%maxit
    ccontrol%factor = fcontrol%factor
    ccontrol%max_col = fcontrol%max_col
    ccontrol%max_sc = fcontrol%max_sc
    ccontrol%indmin = fcontrol%indmin
    ccontrol%valmin = fcontrol%valmin
    ccontrol%itref_max = fcontrol%itref_max
    ccontrol%infeas_check_interval = fcontrol%infeas_check_interval
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%precon = fcontrol%precon
    ccontrol%nsemib = fcontrol%nsemib
    ccontrol%full_max_fill = fcontrol%full_max_fill
    ccontrol%deletion_strategy = fcontrol%deletion_strategy
    ccontrol%restore_problem = fcontrol%restore_problem
    ccontrol%monitor_residuals = fcontrol%monitor_residuals
    ccontrol%cold_start = fcontrol%cold_start
    ccontrol%sif_file_device = fcontrol%sif_file_device

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%feas_tol = fcontrol%feas_tol
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%increase_rho_g_factor = fcontrol%increase_rho_g_factor
    ccontrol%infeas_g_improved_by_factor = fcontrol%infeas_g_improved_by_factor
    ccontrol%increase_rho_b_factor = fcontrol%increase_rho_b_factor
    ccontrol%infeas_b_improved_by_factor = fcontrol%infeas_b_improved_by_factor
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%pivot_tol_for_dependencies = fcontrol%pivot_tol_for_dependencies
    ccontrol%zero_pivot = fcontrol%zero_pivot
    ccontrol%inner_stop_relative = fcontrol%inner_stop_relative
    ccontrol%inner_stop_absolute = fcontrol%inner_stop_absolute
    ccontrol%multiplier_tol = fcontrol%multiplier_tol
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%treat_zero_bounds_as_general                                      &
      = fcontrol%treat_zero_bounds_as_general
    ccontrol%solve_qp = fcontrol%solve_qp
    ccontrol%solve_within_bounds = fcontrol%solve_within_bounds
    ccontrol%randomize = fcontrol%randomize
    ccontrol%array_syntax_worse_than_do_loop                                   &
      = fcontrol%array_syntax_worse_than_do_loop
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file
    ccontrol%each_interval = fcontrol%each_interval

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )

    ! Strings
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( qpa_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_qpa_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_qpa_time_type ), INTENT( IN ) :: ftime
    TYPE ( qpa_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( qpa_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_qpa_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%major_iter = cinform%major_iter
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter
    finform%factorization_status = cinform%factorization_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%nfacts = cinform%nfacts
    finform%nmods = cinform%nmods
    finform%num_g_infeas = cinform%num_g_infeas
    finform%num_b_infeas = cinform%num_b_infeas

    ! Reals
    finform%obj = cinform%obj
    finform%infeas_g = cinform%infeas_g
    finform%infeas_b = cinform%infeas_b
    finform%merit = cinform%merit

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_qpa_inform_type ), INTENT( IN ) :: finform
    TYPE ( qpa_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%major_iter = finform%major_iter
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter
    cinform%factorization_status = finform%factorization_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%nfacts = finform%nfacts
    cinform%nmods = finform%nmods
    cinform%num_g_infeas = finform%num_g_infeas
    cinform%num_b_infeas = finform%num_b_infeas

    ! Reals
    cinform%obj = finform%obj
    cinform%infeas_g = finform%infeas_g
    cinform%infeas_b = finform%infeas_b
    cinform%merit = finform%merit

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_QPA_precision_ciface

!  -------------------------------------
!  C interface to fortran qpa_initialize
!  -------------------------------------

  SUBROUTINE qpa_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( qpa_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_qpa_full_data_type ), POINTER :: fdata
  TYPE ( f_qpa_control_type ) :: fcontrol
  TYPE ( f_qpa_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_qpa_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE qpa_initialize

!  ----------------------------------------
!  C interface to fortran qpa_read_specfile
!  ----------------------------------------

  SUBROUTINE qpa_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( qpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_qpa_control_type ) :: fcontrol
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

  CALL f_qpa_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE qpa_read_specfile

!  ---------------------------------
!  C interface to fortran qpa_inport
!  ---------------------------------

  SUBROUTINE qpa_import( ccontrol, cdata, status, n, m,                        &
                         chtype, hne, hrow, hcol, hptr,                        &
                         catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( qpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_qpa_control_type ) :: fcontrol
  TYPE ( f_qpa_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required QPA structure

  CALL f_qpa_import( fcontrol, fdata, status, n, m,                            &
                     fhtype, hne, hrow, hcol, hptr,                            &
                     fatype, ane, arow, acol, aptr )

!  copy control out

! CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE qpa_import

!  ----------------------------------------
!  C interface to fortran qpa_reset_control
!  ----------------------------------------

  SUBROUTINE qpa_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( qpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpa_control_type ) :: fcontrol
  TYPE ( f_qpa_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_qpa_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE qpa_reset_control

!  -----------------------------------
!  C interface to fortran qpa_solve_qp
!  -----------------------------------

  SUBROUTINE qpa_solve_qp( cdata, status, n, m, hne, hval, g, f, ane, aval,    &
                           cl, cu, xl, xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpa_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_qpa_solve_qp( fdata, status, hval, g, f, aval, cl, cu, xl, xu,        &
                       x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE qpa_solve_qp

!  -------------------------------------
!  C interface to fortran qpa_solve_l1qp
!  -------------------------------------

  SUBROUTINE qpa_solve_l1qp( cdata, status, n, m, hne, hval, g, f,             &
                             rho_g, rho_b, ane, aval, cl, cu, xl, xu,          &
                             x, c, y, z, xstat, cstat) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, rho_g, rho_b
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpa_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_qpa_solve_l1qp( fdata, status, hval, g, f, rho_g, rho_b, aval,        &
                         cl, cu, xl, xu, x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE qpa_solve_l1qp

!  ---------------------------------------
!  C interface to fortran qpa_solve_bcl1qp
!  ---------------------------------------

  SUBROUTINE qpa_solve_bcl1qp( cdata, status, n, m, hne, hval, g, f, rho_g,    &
                               ane, aval, cl, cu, xl, xu, x, c, y, z,          &
                               xstat, cstat ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, rho_g
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_qpa_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_qpa_solve_bcl1qp( fdata, status, hval, g, f, rho_g, aval, cl, cu,    &
                           xl, xu, x, c, y, z, xstat, cstat )
  RETURN

  END SUBROUTINE qpa_solve_bcl1qp

!  --------------------------------------
!  C interface to fortran qpa_information
!  --------------------------------------

  SUBROUTINE qpa_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( qpa_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_qpa_full_data_type ), pointer :: fdata
  TYPE ( f_qpa_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain QPA solution information

  CALL f_qpa_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE qpa_information

!  ------------------------------------
!  C interface to fortran qpa_terminate
!  ------------------------------------

  SUBROUTINE qpa_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_QPA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( qpa_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( qpa_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_qpa_full_data_type ), pointer :: fdata
  TYPE ( f_qpa_control_type ) :: fcontrol
  TYPE ( f_qpa_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_qpa_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE qpa_terminate
