! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  L 2 R T    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. December 19th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_L2RT_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_L2RT_precision, ONLY:                                          &
        f_l2rt_control_type => L2RT_control_type,                              &
        f_l2rt_inform_type => L2RT_inform_type,                                &
        f_l2rt_full_data_type => L2RT_full_data_type,                          &
        f_l2rt_initialize => L2RT_initialize,                                  &
        f_l2rt_read_specfile => L2RT_read_specfile,                            &
        f_l2rt_import_control => L2RT_import_control,                          &
        f_l2rt_solve_problem => L2RT_solve_problem,                            &
        f_l2rt_information => L2RT_information,                                &
        f_l2rt_terminate => L2RT_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: l2rt_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: itmin
      INTEGER ( KIND = ipc_ ) :: itmax
      INTEGER ( KIND = ipc_ ) :: bitmax
      INTEGER ( KIND = ipc_ ) :: extra_vectors
      INTEGER ( KIND = ipc_ ) :: stopping_rule
      INTEGER ( KIND = ipc_ ) :: freq
      REAL ( KIND = rpc_ ) :: stop_relative
      REAL ( KIND = rpc_ ) :: stop_absolute
      REAL ( KIND = rpc_ ) :: fraction_opt
      REAL ( KIND = rpc_ ) :: time_limit
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE l2rt_control_type

    TYPE, BIND( C ) :: l2rt_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: iter_pass2
      INTEGER ( KIND = ipc_ ) :: biters
      INTEGER ( KIND = ipc_ ) :: biter_min
      INTEGER ( KIND = ipc_ ) :: biter_max
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: multiplier
      REAL ( KIND = rpc_ ) :: x_norm
      REAL ( KIND = rpc_ ) :: r_norm
      REAL ( KIND = rpc_ ) :: Atr_norm
      REAL ( KIND = rpc_ ) :: biter_mean
    END TYPE l2rt_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( l2rt_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_l2rt_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%itmin = ccontrol%itmin
    fcontrol%itmax = ccontrol%itmax
    fcontrol%bitmax = ccontrol%bitmax
    fcontrol%extra_vectors = ccontrol%extra_vectors
    fcontrol%stopping_rule = ccontrol%stopping_rule
    fcontrol%freq = ccontrol%freq

    ! Reals
    fcontrol%stop_relative = ccontrol%stop_relative
    fcontrol%stop_absolute = ccontrol%stop_absolute
    fcontrol%fraction_opt = ccontrol%fraction_opt
    fcontrol%time_limit = ccontrol%time_limit

    ! Logicals
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
    TYPE ( f_l2rt_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( l2rt_control_type ), INTENT( OUT ) :: ccontrol
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
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%itmin = fcontrol%itmin
    ccontrol%itmax = fcontrol%itmax
    ccontrol%bitmax = fcontrol%bitmax
    ccontrol%extra_vectors = fcontrol%extra_vectors
    ccontrol%stopping_rule = fcontrol%stopping_rule
    ccontrol%freq = fcontrol%freq

    ! Reals
    ccontrol%stop_relative = fcontrol%stop_relative
    ccontrol%stop_absolute = fcontrol%stop_absolute
    ccontrol%fraction_opt = fcontrol%fraction_opt
    ccontrol%time_limit = fcontrol%time_limit

    ! Logicals
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
    TYPE ( l2rt_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_l2rt_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%iter_pass2 = cinform%iter_pass2
    finform%biters = cinform%biters
    finform%biter_min = cinform%biter_min
    finform%biter_max = cinform%biter_max

    ! Reals
    finform%obj = cinform%obj
    finform%multiplier = cinform%multiplier
    finform%x_norm = cinform%x_norm
    finform%r_norm = cinform%r_norm
    finform%Atr_norm = cinform%Atr_norm
    finform%biter_mean = cinform%biter_mean

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_l2rt_inform_type ), INTENT( IN ) :: finform
    TYPE ( l2rt_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%iter_pass2 = finform%iter_pass2
    cinform%biters = finform%biters
    cinform%biter_min = finform%biter_min
    cinform%biter_max = finform%biter_max

    ! Reals
    cinform%obj = finform%obj
    cinform%multiplier = finform%multiplier
    cinform%x_norm = finform%x_norm
    cinform%r_norm = finform%r_norm
    cinform%Atr_norm = finform%Atr_norm
    cinform%biter_mean = finform%biter_mean

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_L2RT_precision_ciface

!  --------------------------------------
!  C interface to fortran l2rt_initialize
!  --------------------------------------

  SUBROUTINE l2rt_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( l2rt_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_l2rt_full_data_type ), POINTER :: fdata
  TYPE ( f_l2rt_control_type ) :: fcontrol
  TYPE ( f_l2rt_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_l2rt_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE l2rt_initialize

!  -----------------------------------------
!  C interface to fortran l2rt_read_specfile
!  -----------------------------------------

  SUBROUTINE l2rt_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( l2rt_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_l2rt_control_type ) :: fcontrol
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

  CALL f_l2rt_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE l2rt_read_specfile

!  ------------------------------------------
!  C interface to fortran l2rt_import_control
!  ------------------------------------------

  SUBROUTINE l2rt_import_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( l2rt_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_l2rt_control_type ) :: fcontrol
  TYPE ( f_l2rt_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required L2RT structure

   CALL f_l2rt_import_control( fcontrol, fdata, status )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE l2rt_import_control

!  -----------------------------------------
!  C interface to fortran l2rt_solve_problem
!  -----------------------------------------

  SUBROUTINE l2rt_solve_problem( cdata, status, m, n, power, weight, shift,    &
                                 x, u, v ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: power, weight, shift
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: u
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: v
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_l2rt_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_l2rt_solve_problem( fdata, status, m, n, power, weight, shift,        &
                             x, u, v )
  RETURN

  END SUBROUTINE l2rt_solve_problem

!  ---------------------------------------
!  C interface to fortran l2rt_information
!  ---------------------------------------

  SUBROUTINE l2rt_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( l2rt_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_l2rt_full_data_type ), pointer :: fdata
  TYPE ( f_l2rt_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain L2RT solution information

  CALL f_l2rt_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE l2rt_information

!  -------------------------------------
!  C interface to fortran l2rt_terminate
!  -------------------------------------

  SUBROUTINE l2rt_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_L2RT_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( l2rt_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( l2rt_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_l2rt_full_data_type ), pointer :: fdata
  TYPE ( f_l2rt_control_type ) :: fcontrol
  TYPE ( f_l2rt_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_l2rt_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE l2rt_terminate
