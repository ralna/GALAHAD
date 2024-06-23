! THIS VERSION: GALAHAD 4.1 - 2023-06-05 AT 14:00 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  L L S R    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.1. June 5th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_LLSR_precision_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_LLSR_precision, ONLY:                                          &
        f_llsr_control_type => LLSR_control_type,                              &
        f_llsr_time_type => LLSR_time_type,                                    &
        f_llsr_history_type => LLSR_history_type,                              &
        f_llsr_inform_type => LLSR_inform_type,                                &
        f_llsr_full_data_type => LLSR_full_data_type,                          &
        f_llsr_initialize => LLSR_initialize,                                  &
        f_llsr_read_specfile => LLSR_read_specfile,                            &
        f_llsr_import => LLSR_import,                                          &
        f_llsr_import_scaling => LLSR_import_scaling,                          &
        f_llsr_solve_problem => LLSR_solve_problem,                            &
        f_llsr_reset_control => LLSR_reset_control,                            &
        f_llsr_information => LLSR_information,                                &
        f_llsr_terminate => LLSR_terminate

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in => copy_inform_in,                                 &
        copy_sbls_inform_out => copy_inform_out,                               &
        copy_sbls_control_in => copy_control_in,                               &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in => copy_inform_in,                                  &
        copy_sls_inform_out => copy_inform_out,                                &
        copy_sls_control_in => copy_control_in,                                &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_IR_precision_ciface, ONLY:                                     &
        ir_inform_type,                                                        &
        ir_control_type,                                                       &
        copy_ir_inform_in => copy_inform_in,                                   &
        copy_ir_inform_out => copy_inform_out,                                 &
        copy_ir_control_in => copy_control_in,                                 &
        copy_ir_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: llsr_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: new_a
      INTEGER ( KIND = ipc_ ) :: new_s
      INTEGER ( KIND = ipc_ ) :: max_factorizations
      INTEGER ( KIND = ipc_ ) :: taylor_max_degree
      REAL ( KIND = rpc_ ) :: initial_multiplier
      REAL ( KIND = rpc_ ) :: lower
      REAL ( KIND = rpc_ ) :: upper
      REAL ( KIND = rpc_ ) :: stop_normal
      LOGICAL ( KIND = C_BOOL ) :: use_initial_multiplier
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: definite_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( ir_control_type ) :: ir_control
    END TYPE llsr_control_type

    TYPE, BIND( C ) :: llsr_time_type
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
    END TYPE llsr_time_type

    TYPE, BIND( C ) :: llsr_history_type
      REAL ( KIND = rpc_ ) :: lambda
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: r_norm
    END TYPE llsr_history_type

    TYPE, BIND( C ) :: llsr_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: factorizations
      INTEGER ( KIND = ipc_ ) :: len_history
      REAL ( KIND = rpc_ ) :: r_norm
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: multiplier
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( llsr_time_type ) :: time
      TYPE ( llsr_history_type ), DIMENSION( 100 ) :: history
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( ir_inform_type ) :: ir_inform
    END TYPE llsr_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( llsr_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_llsr_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ipc_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%new_a = ccontrol%new_a
    fcontrol%new_s = ccontrol%new_s
    fcontrol%max_factorizations = ccontrol%max_factorizations
    fcontrol%taylor_max_degree = ccontrol%taylor_max_degree

    ! Reals
    fcontrol%initial_multiplier = ccontrol%initial_multiplier
    fcontrol%lower = ccontrol%lower
    fcontrol%upper = ccontrol%upper
    fcontrol%stop_normal = ccontrol%stop_normal

    ! Logicals
    fcontrol%use_initial_multiplier = ccontrol%use_initial_multiplier
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_ir_control_in( ccontrol%ir_control, fcontrol%ir_control )

    ! Strings
    DO i = 1, LEN( fcontrol%definite_linear_solver )
      IF ( ccontrol%definite_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%definite_linear_solver( i : i )                                &
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
    TYPE ( f_llsr_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( llsr_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ipc_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%new_a = fcontrol%new_a
    ccontrol%new_s = fcontrol%new_s
    ccontrol%max_factorizations = fcontrol%max_factorizations
    ccontrol%taylor_max_degree = fcontrol%taylor_max_degree

    ! Reals
    ccontrol%initial_multiplier = fcontrol%initial_multiplier
    ccontrol%lower = fcontrol%lower
    ccontrol%upper = fcontrol%upper
    ccontrol%stop_normal = fcontrol%stop_normal

    ! Logicals
    ccontrol%use_initial_multiplier = fcontrol%use_initial_multiplier
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_ir_control_out( fcontrol%ir_control, ccontrol%ir_control )

    ! Strings
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
    TYPE ( llsr_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_llsr_time_type ), INTENT( OUT ) :: ftime

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
    TYPE ( f_llsr_time_type ), INTENT( IN ) :: ftime
    TYPE ( llsr_time_type ), INTENT( OUT ) :: ctime

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
    TYPE ( llsr_history_type ), INTENT( IN ), DIMENSION( 100 ) :: chistory
    TYPE ( f_llsr_history_type ), INTENT( OUT ), DIMENSION( 100 ) :: fhistory

    ! Reals
    fhistory%lambda = chistory%lambda
    fhistory%x_norm = chistory%x_norm
    fhistory%r_norm = chistory%r_norm
    RETURN

    END SUBROUTINE copy_history_in

!  copy fortran history parameters to C

    SUBROUTINE copy_history_out( fhistory, chistory )
    TYPE ( f_llsr_history_type ), INTENT( IN ), DIMENSION( 100 ) :: fhistory
    TYPE ( llsr_history_type ), INTENT( OUT ), DIMENSION( 100 ) :: chistory

    ! Reals
    chistory%lambda = fhistory%lambda
    chistory%x_norm = fhistory%x_norm
    chistory%r_norm = fhistory%r_norm
    RETURN

    END SUBROUTINE copy_history_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( llsr_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_llsr_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ipc_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%factorizations = cinform%factorizations
    finform%len_history = cinform%len_history

    ! Reals
    finform%r_norm = cinform%r_norm
    finform%x_norm = cinform%x_norm
    finform%multiplier = cinform%multiplier

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_history_in( cinform%history, finform%history )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
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
    TYPE ( f_llsr_inform_type ), INTENT( IN ) :: finform
    TYPE ( llsr_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ipc_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%factorizations = finform%factorizations
    cinform%len_history = finform%len_history

    ! Reals
    cinform%r_norm = finform%r_norm
    cinform%x_norm = finform%x_norm
    cinform%multiplier = finform%multiplier

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_history_out( finform%history, cinform%history )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
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

  END MODULE GALAHAD_LLSR_precision_ciface

!  -------------------------------------
!  C interface to fortran llsr_initialize
!  -------------------------------------

  SUBROUTINE llsr_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( llsr_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_llsr_full_data_type ), POINTER :: fdata
  TYPE ( f_llsr_control_type ) :: fcontrol
  TYPE ( f_llsr_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_llsr_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE llsr_initialize

!  ----------------------------------------
!  C interface to fortran llsr_read_specfile
!  ----------------------------------------

  SUBROUTINE llsr_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( llsr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_llsr_control_type ) :: fcontrol
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

  CALL f_llsr_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE llsr_read_specfile

!  ---------------------------------
!  C interface to fortran llsr_inport
!  ---------------------------------

  SUBROUTINE llsr_import( ccontrol, cdata, status, m, n,                       &
                          catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( llsr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_llsr_control_type ) :: fcontrol
  TYPE ( f_llsr_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required LLSR structure

  CALL f_llsr_import( fcontrol, fdata, status, m, n,                           &
                      fatype, ane, arow, acol, aptr )

!  copy control out

! CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE llsr_import

!  ------------------------------------------
!  C interface to fortran llsr_inport_scaling
!  ------------------------------------------

  SUBROUTINE llsr_import_scaling( ccontrol, cdata, status, n,                  &
                                  cstype, sne, srow, scol, sptr ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( llsr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, sne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( sne ), OPTIONAL :: srow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( sne ), OPTIONAL :: scol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: sptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cstype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cstype ) ) :: fstype
  TYPE ( f_llsr_control_type ) :: fcontrol
  TYPE ( f_llsr_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )
!write(6,*) ' out ', fcontrol%out
!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )
!write(6,*) ' out from data', fdata%llsr_control%out
!write(6,*) ' out from data', fdata%LLSR_data%control%out

!  convert C string to Fortran string

  fstype = cstr_to_fchar( cstype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required LLSR structure

  CALL f_llsr_import_scaling( fdata, status, fstype, sne, srow, scol, sptr )

!  copy control out

! CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE llsr_import_scaling

!  ---------------------------------------
!  C interface to fortran llsr_reset_control
!  ----------------------------------------

  SUBROUTINE llsr_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( llsr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_llsr_control_type ) :: fcontrol
  TYPE ( f_llsr_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_LLSR_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE llsr_reset_control

!  ----------------------------------------
!  C interface to fortran llsr_solve_problem
!  ----------------------------------------

  SUBROUTINE llsr_solve_problem( cdata, status, m, n, power, weight, ane,      &
                                 aval, b, x, sne, sval ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n, ane, sne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), OPTIONAL, INTENT( IN ), DIMENSION( sne ) :: sval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: b
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: power, weight
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_llsr_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )
!write(6,*) ' out from data solve ', fdata%llsr_control%out
!write(6,*) ' out from data solve ', fdata%llsr_data%control%out

!  solve the trust-region problem

  CALL f_llsr_solve_problem( fdata, status, power, weight, aval, b, x, sval )
  RETURN

  END SUBROUTINE llsr_solve_problem

!  --------------------------------------
!  C interface to fortran llsr_information
!  --------------------------------------

  SUBROUTINE llsr_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( llsr_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_llsr_full_data_type ), pointer :: fdata
  TYPE ( f_llsr_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain LLSR solution information

  CALL f_llsr_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE llsr_information

!  ------------------------------------
!  C interface to fortran llsr_terminate
!  ------------------------------------

  SUBROUTINE llsr_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_LLSR_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( llsr_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( llsr_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_llsr_full_data_type ), pointer :: fdata
  TYPE ( f_llsr_control_type ) :: fcontrol
  TYPE ( f_llsr_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_llsr_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE llsr_terminate
