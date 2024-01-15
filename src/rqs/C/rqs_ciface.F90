! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  R Q S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. December 12th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_RQS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_RQS_precision, ONLY:                                           &
        f_rqs_control_type => RQS_control_type,                                &
        f_rqs_time_type => RQS_time_type,                                      &
        f_rqs_history_type => RQS_history_type,                                &
        f_rqs_inform_type => RQS_inform_type,                                  &
        f_rqs_full_data_type => RQS_full_data_type,                            &
        f_rqs_initialize => RQS_initialize,                                    &
        f_rqs_read_specfile => RQS_read_specfile,                              &
        f_rqs_import => RQS_import,                                            &
        f_rqs_import_m => RQS_import_m,                                        &
        f_rqs_import_a => RQS_import_a,                                        &
        f_rqs_solve_problem => RQS_solve_problem,                              &
        f_rqs_reset_control => RQS_reset_control,                              &
        f_rqs_information => RQS_information,                                  &
        f_rqs_terminate => RQS_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in => copy_inform_in,                                  &
        copy_sls_inform_out => copy_inform_out,                                &
        copy_sls_control_in => copy_control_in,                                &
        copy_sls_control_out => copy_control_out

     USE GALAHAD_IR_precision_ciface, ONLY:                                    &
         ir_inform_type,                                                       &
         ir_control_type,                                                      &
         copy_ir_inform_in => copy_inform_in,                                  &
         copy_ir_inform_out => copy_inform_out,                                &
         copy_ir_control_in => copy_control_in,                                &
         copy_ir_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: rqs_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: problem
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: dense_factorization
      INTEGER ( KIND = ipc_ ) :: new_h
      INTEGER ( KIND = ipc_ ) :: new_m
      INTEGER ( KIND = ipc_ ) :: new_a
      INTEGER ( KIND = ipc_ ) :: max_factorizations
      INTEGER ( KIND = ipc_ ) :: inverse_itmax
      INTEGER ( KIND = ipc_ ) :: taylor_max_degree
      REAL ( KIND = rpc_ ) :: initial_multiplier
      REAL ( KIND = rpc_ ) :: lower
      REAL ( KIND = rpc_ ) :: upper
      REAL ( KIND = rpc_ ) :: stop_normal
      REAL ( KIND = rpc_ ) :: stop_hard
      REAL ( KIND = rpc_ ) :: start_invit_tol
      REAL ( KIND = rpc_ ) :: start_invitmax_tol
      LOGICAL ( KIND = C_BOOL ) :: use_initial_multiplier
      LOGICAL ( KIND = C_BOOL ) :: initialize_approx_eigenvector
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: problem_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: definite_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( ir_control_type ) :: ir_control
    END TYPE rqs_control_type

    TYPE, BIND( C ) :: rqs_time_type
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
    END TYPE rqs_time_type

    TYPE, BIND( C ) :: rqs_history_type
      REAL ( KIND = rpc_ ) :: lambda
      REAL ( KIND = rpc_ ) :: x_norm
    END TYPE rqs_history_type

    TYPE, BIND( C ) :: rqs_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: factorizations
      INTEGER ( KIND = longc_ ) :: max_entries_factors
      INTEGER ( KIND = ipc_ ) :: len_history
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: obj_regularized
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: pole
      LOGICAL ( KIND = C_BOOL ) :: dense_factorization
      LOGICAL ( KIND = C_BOOL ) :: hard_case
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( rqs_time_type ) :: time
      TYPE ( rqs_history_type ), DIMENSION( 100 ) :: history
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( ir_inform_type ) :: ir_inform
    END TYPE rqs_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( rqs_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_rqs_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%problem = ccontrol%problem
    fcontrol%print_level = ccontrol%print_level
    fcontrol%dense_factorization = ccontrol%dense_factorization
    fcontrol%new_h = ccontrol%new_h
    fcontrol%new_m = ccontrol%new_m
    fcontrol%new_a = ccontrol%new_a
    fcontrol%max_factorizations = ccontrol%max_factorizations
    fcontrol%inverse_itmax = ccontrol%inverse_itmax
    fcontrol%taylor_max_degree = ccontrol%taylor_max_degree

    ! Reals
    fcontrol%initial_multiplier = ccontrol%initial_multiplier
    fcontrol%lower = ccontrol%lower
    fcontrol%upper = ccontrol%upper
    fcontrol%stop_normal = ccontrol%stop_normal
    fcontrol%stop_hard = ccontrol%stop_hard
    fcontrol%start_invit_tol = ccontrol%start_invit_tol
    fcontrol%start_invitmax_tol = ccontrol%start_invitmax_tol

    ! Logicals
    fcontrol%use_initial_multiplier = ccontrol%use_initial_multiplier
    fcontrol%initialize_approx_eigenvector                                     &
      = ccontrol%initialize_approx_eigenvector
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_ir_control_in( ccontrol%ir_control, fcontrol%ir_control )

    ! Strings
    DO i = 1, LEN( fcontrol%problem_file )
      IF ( ccontrol%problem_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%problem_file( i : i ) = ccontrol%problem_file( i )
    END DO
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i )                                &
        = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%definite_linear_solver )
      IF ( ccontrol%definite_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%definite_linear_solver( i : i )                                 &
        = ccontrol%definite_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_rqs_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( rqs_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%problem = fcontrol%problem
    ccontrol%print_level = fcontrol%print_level
    ccontrol%dense_factorization = fcontrol%dense_factorization
    ccontrol%new_h = fcontrol%new_h
    ccontrol%new_m = fcontrol%new_m
    ccontrol%new_a = fcontrol%new_a
    ccontrol%max_factorizations = fcontrol%max_factorizations
    ccontrol%inverse_itmax = fcontrol%inverse_itmax
    ccontrol%taylor_max_degree = fcontrol%taylor_max_degree

    ! Reals
    ccontrol%initial_multiplier = fcontrol%initial_multiplier
    ccontrol%lower = fcontrol%lower
    ccontrol%upper = fcontrol%upper
    ccontrol%stop_normal = fcontrol%stop_normal
    ccontrol%stop_hard = fcontrol%stop_hard
    ccontrol%start_invit_tol = fcontrol%start_invit_tol
    ccontrol%start_invitmax_tol = fcontrol%start_invitmax_tol

    ! Logicals
    ccontrol%use_initial_multiplier = fcontrol%use_initial_multiplier
    ccontrol%initialize_approx_eigenvector                                     &
      = fcontrol%initialize_approx_eigenvector
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_ir_control_out( fcontrol%ir_control, ccontrol%ir_control )

    ! Strings
    l = LEN( fcontrol%problem_file )
    DO i = 1, l
      ccontrol%problem_file( i ) = fcontrol%problem_file( i : i )
    END DO
    ccontrol%problem_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%definite_linear_solver )
    DO i = 1, l
      ccontrol%definite_linear_solver( i )                                     &
        = fcontrol%definite_linear_solver( i : i )
    END DO
    ccontrol%definite_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( rqs_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_rqs_time_type ), INTENT( OUT ) :: ftime

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
    TYPE ( f_rqs_time_type ), INTENT( IN ) :: ftime
    TYPE ( rqs_time_type ), INTENT( OUT ) :: ctime

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

!  copy C history parameters to fortran

    SUBROUTINE copy_history_in( chistory, fhistory )
    TYPE ( rqs_history_type ), INTENT( IN ), DIMENSION( 100 ) :: chistory
    TYPE ( f_rqs_history_type ), INTENT( OUT ), DIMENSION( 100 ) :: fhistory

    ! Reals
    fhistory%lambda = chistory%lambda
    fhistory%x_norm = chistory%x_norm
    RETURN

    END SUBROUTINE copy_history_in

!  copy fortran history parameters to C

    SUBROUTINE copy_history_out( fhistory, chistory )
    TYPE ( f_rqs_history_type ), INTENT( IN ), DIMENSION( 100 ) :: fhistory
    TYPE ( rqs_history_type ), INTENT( OUT ), DIMENSION( 100 ) :: chistory

    ! Reals
    chistory%lambda = fhistory%lambda
    chistory%x_norm = fhistory%x_norm
    RETURN

    END SUBROUTINE copy_history_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( rqs_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_rqs_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%factorizations = cinform%factorizations
    finform%max_entries_factors = cinform%max_entries_factors
    finform%len_history = cinform%len_history

    ! Reals
    finform%obj = cinform%obj
    finform%obj_regularized = cinform%obj_regularized
    finform%x_norm = cinform%x_norm
    finform%multiplier = cinform%multiplier
    finform%pole = cinform%pole

    ! Logicals
    finform%dense_factorization = cinform%dense_factorization
    finform%hard_case = cinform%hard_case

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_history_in( cinform%history, finform%history )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_ir_inform_in( cinform%ir_inform, finform%ir_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_rqs_inform_type ), INTENT( IN ) :: finform
    TYPE ( rqs_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%factorizations = finform%factorizations
    cinform%max_entries_factors = finform%max_entries_factors
    cinform%len_history = finform%len_history

    ! Reals
    cinform%obj = finform%obj
    cinform%obj_regularized = finform%obj_regularized
    cinform%x_norm = finform%x_norm
    cinform%multiplier = finform%multiplier
    cinform%pole = finform%pole

    ! Logicals
    cinform%dense_factorization = finform%dense_factorization
    cinform%hard_case = finform%hard_case

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_history_out( finform%history, cinform%history )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_ir_inform_out( finform%ir_inform, cinform%ir_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_RQS_precision_ciface

!  -------------------------------------
!  C interface to fortran rqs_initialize
!  -------------------------------------

  SUBROUTINE rqs_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( rqs_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_rqs_full_data_type ), POINTER :: fdata
  TYPE ( f_rqs_control_type ) :: fcontrol
  TYPE ( f_rqs_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_rqs_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE rqs_initialize

!  ----------------------------------------
!  C interface to fortran rqs_read_specfile
!  ----------------------------------------

  SUBROUTINE rqs_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( rqs_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_rqs_control_type ) :: fcontrol
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

  CALL f_rqs_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE rqs_read_specfile

!  ---------------------------------
!  C interface to fortran rqs_inport
!  ---------------------------------

  SUBROUTINE rqs_import( ccontrol, cdata, status, n,                           &
                         chtype, hne, hrow, hcol, hptr  ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( rqs_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_rqs_control_type ) :: fcontrol
  TYPE ( f_rqs_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required RQS structure

  CALL f_rqs_import( fcontrol, fdata, status, n,                               &
                     fhtype, hne, hrow, hcol, hptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE rqs_import

!  -----------------------------------
!  C interface to fortran rqs_inport_m
!  -----------------------------------

  SUBROUTINE rqs_import_m( cdata, status, n, cmtype, mne, mrow, mcol,          &
                           mptr ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, mne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( mne ), OPTIONAL :: mrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( mne ), OPTIONAL :: mcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: mptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cmtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cmtype ) ) :: fmtype
  TYPE ( f_rqs_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  convert C string to Fortran string

  fmtype = cstr_to_fchar( cmtype )

!  import the problem data into the required RQS structure

  CALL f_rqs_import_m( fdata, status, fmtype, mne, mrow, mcol, mptr )
  RETURN

  END SUBROUTINE rqs_import_m

!  -----------------------------------
!  C interface to fortran rqs_inport_a
!  -----------------------------------

  SUBROUTINE rqs_import_a( cdata, status, m, caytpe, ane, arow, acol,          &
                           aptr ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: caytpe

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( caytpe ) ) :: faytpe
  TYPE ( f_rqs_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  convert C string to Fortran string

  faytpe = cstr_to_fchar( caytpe )

  CALL f_rqs_import_a( fdata, status, m, faytpe, ane, arow, acol, aptr )
  RETURN

  END SUBROUTINE rqs_import_a

!  ----------------------------------------
!  C interface to fortran rqs_reset_control
!  ----------------------------------------

  SUBROUTINE rqs_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( rqs_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rqs_control_type ) :: fcontrol
  TYPE ( f_rqs_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_rqs_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE rqs_reset_control

!  ----------------------------------------
!  C interface to fortran rqs_solve_problem
!  ----------------------------------------

  SUBROUTINE rqs_solve_problem( cdata, status, n, power, weight,               &
                                f, c, hne, hval, x,                            &
                                mne, mval, m, ane, aval, y ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, mne, ane
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( mne ) :: mval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: power, weight, f
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rqs_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_RQS_solve_problem( fdata, status, power, weight, f, c, hval, x,       &
                            mval, aval, y )
  RETURN

  END SUBROUTINE rqs_solve_problem

!  --------------------------------------
!  C interface to fortran rqs_information
!  --------------------------------------

  SUBROUTINE rqs_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( rqs_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_rqs_full_data_type ), pointer :: fdata
  TYPE ( f_rqs_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain RQS solution information

  CALL f_rqs_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE rqs_information

!  ------------------------------------
!  C interface to fortran rqs_terminate
!  ------------------------------------

  SUBROUTINE rqs_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_RQS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( rqs_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( rqs_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_rqs_full_data_type ), pointer :: fdata
  TYPE ( f_rqs_control_type ) :: fcontrol
  TYPE ( f_rqs_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_rqs_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE rqs_terminate
