! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  D P S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_DPS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_DPS_precision, ONLY:                                           &
        f_dps_control_type       => DPS_control_type,                          &
        f_dps_time_type          => DPS_time_type,                             &
        f_dps_inform_type        => DPS_inform_type,                           &
        f_dps_full_data_type     => DPS_full_data_type,                        &
        f_dps_initialize         => DPS_initialize,                            &
        f_dps_read_specfile      => DPS_read_specfile,                         &
        f_dps_import             => DPS_import,                                &
        f_dps_reset_control      => DPS_reset_control,                         &
        f_dps_solve_tr_problem   => DPS_solve_tr_problem,                      &
        f_dps_solve_rq_problem   => DPS_solve_rq_problem,                      &
        f_dps_resolve_tr_problem => DPS_resolve_tr_problem,                    &
        f_dps_resolve_rq_problem => DPS_resolve_rq_problem,                    &
        f_dps_information        => DPS_information,                           &
        f_dps_terminate          => DPS_terminate

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

    TYPE, BIND( C ) :: dps_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: problem
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: new_h
      INTEGER ( KIND = ipc_ ) :: taylor_max_degree
      REAL ( KIND = rpc_ ) :: eigen_min
      REAL ( KIND = rpc_ ) :: lower
      REAL ( KIND = rpc_ ) :: upper
      REAL ( KIND = rpc_ ) :: stop_normal
      REAL ( KIND = rpc_ ) :: stop_absolute_normal
      LOGICAL ( KIND = C_BOOL ) :: goldfarb
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: problem_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
    END TYPE dps_control_type

    TYPE, BIND( C ) :: dps_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE dps_time_type

    TYPE, BIND( C ) :: dps_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: mod_1by1
      INTEGER ( KIND = ipc_ ) :: mod_2by2
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: obj_regularized
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: pole
      LOGICAL ( KIND = C_BOOL ) :: hard_case
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( dps_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
    END TYPE dps_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( dps_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_dps_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%problem = ccontrol%problem
    fcontrol%print_level = ccontrol%print_level
    fcontrol%new_h = ccontrol%new_h
    fcontrol%taylor_max_degree = ccontrol%taylor_max_degree

    ! Reals
    fcontrol%eigen_min = ccontrol%eigen_min
    fcontrol%lower = ccontrol%lower
    fcontrol%upper = ccontrol%upper
    fcontrol%stop_normal = ccontrol%stop_normal
    fcontrol%stop_absolute_normal = ccontrol%stop_absolute_normal

    ! Logicals
    fcontrol%goldfarb = ccontrol%goldfarb
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )

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
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_dps_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( dps_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%problem = fcontrol%problem
    ccontrol%print_level = fcontrol%print_level
    ccontrol%new_h = fcontrol%new_h
    ccontrol%taylor_max_degree = fcontrol%taylor_max_degree

    ! Reals
    ccontrol%eigen_min = fcontrol%eigen_min
    ccontrol%lower = fcontrol%lower
    ccontrol%upper = fcontrol%upper
    ccontrol%stop_normal = fcontrol%stop_normal
    ccontrol%stop_absolute_normal = fcontrol%stop_absolute_normal

    ! Logicals
    ccontrol%goldfarb = fcontrol%goldfarb
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )

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
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( dps_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_dps_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_dps_time_type ), INTENT( IN ) :: ftime
    TYPE ( dps_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( dps_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_dps_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%mod_1by1 = cinform%mod_1by1
    finform%mod_2by2 = cinform%mod_2by2

    ! Reals
    finform%obj = cinform%obj
    finform%obj_regularized = cinform%obj_regularized
    finform%x_norm = cinform%x_norm
    finform%multiplier = cinform%multiplier
    finform%pole = cinform%pole

    ! Logicals
    finform%hard_case = cinform%hard_case

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
    TYPE ( f_dps_inform_type ), INTENT( IN ) :: finform
    TYPE ( dps_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%mod_1by1 = finform%mod_1by1
    cinform%mod_2by2 = finform%mod_2by2

    ! Reals
    cinform%obj = finform%obj
    cinform%obj_regularized = finform%obj_regularized
    cinform%x_norm = finform%x_norm
    cinform%multiplier = finform%multiplier
    cinform%pole = finform%pole

    ! Logicals
    cinform%hard_case = finform%hard_case

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

  END MODULE GALAHAD_DPS_precision_ciface

!  -------------------------------------
!  C interface to fortran dps_initialize
!  -------------------------------------

  SUBROUTINE dps_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( dps_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_dps_full_data_type ), POINTER :: fdata
  TYPE ( f_dps_control_type ) :: fcontrol
  TYPE ( f_dps_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_dps_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dps_initialize

!  ----------------------------------------
!  C interface to fortran dps_read_specfile
!  ----------------------------------------

  SUBROUTINE dps_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( dps_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_dps_control_type ) :: fcontrol
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

  CALL f_dps_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dps_read_specfile

!  ---------------------------------
!  C interface to fortran dps_inport
!  ---------------------------------

  SUBROUTINE dps_import( ccontrol, cdata, status, n,                           &
                         chtype, hne, hrow, hcol, hptr  ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( dps_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_dps_control_type ) :: fcontrol
  TYPE ( f_dps_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required DPS structure

  CALL f_dps_import( fcontrol, fdata, status, n,                               &
                     fhtype, hne, hrow, hcol, hptr )
  RETURN

  END SUBROUTINE dps_import

!  ---------------------------------------
!  C interface to fortran dps_reset_control
!  ----------------------------------------

  SUBROUTINE dps_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( dps_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dps_control_type ) :: fcontrol
  TYPE ( f_dps_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_DPS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE dps_reset_control

!  -------------------------------------------
!  C interface to fortran dps_solve_tr_problem
!  -------------------------------------------

  SUBROUTINE dps_solve_tr_problem( cdata, status, n, hne, hval, c, f,          &
                                   radius, x ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, radius
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dps_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dps_solve_tr_problem( fdata, status, hval, c, f, radius, x )
  RETURN

  END SUBROUTINE dps_solve_tr_problem

!  -------------------------------------------
!  C interface to fortran dps_solve_rq_problem
!  -------------------------------------------

  SUBROUTINE dps_solve_rq_problem( cdata, status, n, hne, hval, c, f,          &
                                   weight, power, x ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, weight, power
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dps_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dps_solve_rq_problem( fdata, status, hval, c, f, weight, power, x )
  RETURN

  END SUBROUTINE dps_solve_rq_problem

!  ---------------------------------------------
!  C interface to fortran dps_resolve_tr_problem
!  ---------------------------------------------

  SUBROUTINE dps_resolve_tr_problem( cdata, status, n,                         &
                                     c, f, radius, x ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, radius
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dps_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dps_resolve_tr_problem( fdata, status, c, f, radius, x )
  RETURN

  END SUBROUTINE dps_resolve_tr_problem

!  ---------------------------------------------
!  C interface to fortran dps_resolve_rq_problem
!  ---------------------------------------------

  SUBROUTINE dps_resolve_rq_problem( cdata, status, n, c, f,                   &
                                     weight, power, x ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f, weight, power
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dps_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_dps_resolve_rq_problem( fdata, status, c, f, weight, power, x )
  RETURN

  END SUBROUTINE dps_resolve_rq_problem

!  --------------------------------------
!  C interface to fortran dps_information
!  --------------------------------------

  SUBROUTINE dps_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dps_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_dps_full_data_type ), pointer :: fdata
  TYPE ( f_dps_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain DPS solution information

  CALL f_dps_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE dps_information

!  ------------------------------------
!  C interface to fortran dps_terminate
!  ------------------------------------

  SUBROUTINE dps_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_DPS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dps_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( dps_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_dps_full_data_type ), pointer :: fdata
  TYPE ( f_dps_control_type ) :: fcontrol
  TYPE ( f_dps_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_dps_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE dps_terminate
