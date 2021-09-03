! THIS VERSION: GALAHAD 3.3 - 03/08/2021 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ U G O   C   I N T E R F A C E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. July 28th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_UGO_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_UGO_double, only:                                              &
        f_ugo_time_type      => UGO_time_type,                                 &
        f_ugo_inform_type    => UGO_inform_type,                               &
        f_ugo_control_type   => UGO_control_type,                              &
        f_ugo_full_data_type => UGO_full_data_type,                            &
        f_ugo_initialize     => UGO_initialize,                                &
        f_ugo_read_specfile  => UGO_read_specfile,                             &
        f_ugo_import         => UGO_import,                                    &
        f_ugo_solve_reverse  => UGO_solve_reverse,                             &
        f_ugo_solve_direct   => UGO_solve_direct,                              &
        f_ugo_information    => UGO_information,                               &
        f_ugo_terminate      => UGO_terminate
    USE GALAHAD_NLPT_double, only:                                             &
        f_nlpt_userdata_type => NLPT_userdata_type

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ugo_control_type
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: start_print
      INTEGER ( KIND = C_INT ) :: stop_print
      INTEGER ( KIND = C_INT ) :: print_gap
      INTEGER ( KIND = C_INT ) :: maxit
      INTEGER ( KIND = C_INT ) :: initial_points
      INTEGER ( KIND = C_INT ) :: storage_increment 
      INTEGER ( KIND = C_INT ) :: buffer
      INTEGER ( KIND = C_INT ) :: lipschitz_estimate_used
      INTEGER ( KIND = C_INT ) :: next_interval_selection 
      INTEGER ( KIND = C_INT ) :: refine_with_newton 
      INTEGER ( KIND = C_INT ) :: alive_unit 
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      REAL ( KIND = wp ) :: stop_length 
      REAL ( KIND = wp ) :: small_g_for_newton 
      REAL ( KIND = wp ) :: small_g 
      REAL ( KIND = wp ) :: obj_sufficient 
      REAL ( KIND = wp ) :: global_lipschitz_constant 
      REAL ( KIND = wp ) :: reliability_parameter 
      REAL ( KIND = wp ) :: lipschitz_lower_bound 
      REAL ( KIND = wp ) :: cpu_time_limit 
      REAL ( KIND = wp ) :: clock_time_limit 
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix 
    END TYPE ugo_control_type

    TYPE, BIND( C ) :: ugo_time_type
      REAL ( KIND = sp ) :: total
      REAL ( KIND = wp ) :: clock_total 
    END TYPE ugo_time_type

    TYPE, BIND( C ) :: ugo_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: eval_status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: f_eval
      INTEGER ( KIND = C_INT ) :: g_eval
      INTEGER ( KIND = C_INT ) :: h_eval 
      TYPE ( ugo_time_type ) :: time
    END TYPE ugo_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_fgh( x, f, g, h, userdata ) result( status ) BIND( C )
        USE iso_c_binding
        IMPORT :: wp
        REAL ( KIND = wp ), INTENT( IN ), value :: x
        REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h
        TYPE ( C_PTR ), INTENT( IN ), value :: userdata
        INTEGER ( KIND = C_INT ) :: status
      END FUNCTION eval_fgh
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol ) 
    TYPE ( ugo_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ugo_control_type ), INTENT( OUT ) :: fcontrol
    INTEGER :: i
    
    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%initial_points = ccontrol%initial_points
    fcontrol%storage_increment  = ccontrol%storage_increment 
    fcontrol%buffer = ccontrol%buffer
    fcontrol%lipschitz_estimate_used = ccontrol%lipschitz_estimate_used
    fcontrol%next_interval_selection = ccontrol%next_interval_selection
    fcontrol%refine_with_newton = ccontrol%refine_with_newton
    fcontrol%alive_unit = ccontrol%alive_unit

    ! Reals
    fcontrol%stop_length = ccontrol%stop_length 
    fcontrol%small_g_for_newton = ccontrol%small_g_for_newton
    fcontrol%small_g = ccontrol%small_g
    fcontrol%obj_sufficient = ccontrol%obj_sufficient
    fcontrol%global_lipschitz_constant = ccontrol%global_lipschitz_constant
    fcontrol%reliability_parameter = ccontrol%reliability_parameter
    fcontrol%lipschitz_lower_bound = ccontrol%lipschitz_lower_bound
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit 

    ! logicals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

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

    SUBROUTINE copy_control_out( fcontrol, ccontrol ) 
    TYPE ( f_ugo_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( ugo_control_type ), INTENT( OUT ) :: ccontrol
    INTEGER :: i
    
    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%initial_points = fcontrol%initial_points
    ccontrol%storage_increment  = fcontrol%storage_increment 
    ccontrol%buffer = fcontrol%buffer
    ccontrol%lipschitz_estimate_used = fcontrol%lipschitz_estimate_used
    ccontrol%next_interval_selection = fcontrol%next_interval_selection
    ccontrol%refine_with_newton = fcontrol%refine_with_newton
    ccontrol%alive_unit = fcontrol%alive_unit

    ! Reals
    ccontrol%stop_length = fcontrol%stop_length 
    ccontrol%small_g_for_newton = fcontrol%small_g_for_newton
    ccontrol%small_g = fcontrol%small_g
    ccontrol%obj_sufficient = fcontrol%obj_sufficient
    ccontrol%global_lipschitz_constant = fcontrol%global_lipschitz_constant
    ccontrol%reliability_parameter = fcontrol%reliability_parameter
    ccontrol%lipschitz_lower_bound = fcontrol%lipschitz_lower_bound
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

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

!  copy C information parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( ugo_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_ugo_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%f_eval = cinform%f_eval
    finform%g_eval = cinform%g_eval
    finform%h_eval = cinform%h_eval
    
    ! Time derived type
    finform%time%total = cinform%time%total
    finform%time%clock_total = cinform%time%clock_total

    ! Strings
    DO i = 1, 81
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_ugo_inform_type ), INTENT( IN ) :: finform
    TYPE ( ugo_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i

    ! integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%f_eval = finform%f_eval
    cinform%g_eval = finform%g_eval
    cinform%h_eval = finform%h_eval
    
    ! Time derived type
    cinform%time%total = finform%time%total
    cinform%time%clock_total = finform%time%clock_total

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( LEN( finform%bad_alloc ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_UGO_double_ciface

!  -------------------------------------
!  C interface to fortran ugo_initialize
!  -------------------------------------

  SUBROUTINE ugo_initialize( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( ugo_control_type ), INTENT( OUT ) :: ccontrol
  TYPE ( ugo_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_ugo_full_data_type ), POINTER :: fdata
  TYPE ( f_ugo_control_type ) :: fcontrol
  TYPE ( f_ugo_inform_type ) :: finform

!  allocate fdata 

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_ugo_initialize( fdata, fcontrol, finform ) 

!  initialize eval_status (for reverse communication INTERFACE)

  cinform%eval_status = 0

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE ugo_initialize

!  ----------------------------------------
!  C interface to fortran ugo_read_specfile
!  ----------------------------------------

  SUBROUTINE ugo_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( ugo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), value :: cspecfile

!  local variables

  TYPE ( f_ugo_control_type ) :: fcontrol
  CHARACTER( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile

!  device unit number for specfile

  INTEGER ( KIND = C_INT ), parameter :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  Copy control in

  CALL copy_control_in( ccontrol, fcontrol )
  
!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )
  
!  read control parameters from the specfile

  CALL f_ugo_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol )
  RETURN

  END SUBROUTINE ugo_read_specfile

!  ---------------------------------
!  C interface to fortran ugo_inport
!  ---------------------------------

  SUBROUTINE ugo_import( ccontrol, cdata, status, xl, xu ) BIND( C )
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  REAL ( KIND = wp ), INTENT( IN ) :: xl, xu
  TYPE ( ugo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_ugo_control_type ) :: fcontrol
  TYPE ( f_ugo_full_data_type ), POINTER :: fdata

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  import the problem data into the required UGO structure

  CALL f_ugo_import( fcontrol, fdata, status, xl, xu )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol )
  RETURN

  END SUBROUTINE ugo_import

!  ---------------------------------------
!  C interface to fortran ugo_solve_direct
!  ---------------------------------------

  SUBROUTINE ugo_solve_direct( cdata, cuserdata, status, x, f, g, h,           &
                               ceval_fgh ) BIND( C ) 
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  REAL ( KIND = wp ), INTENT( INOUT ) :: x, f, g, h
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_fgh

!  local variables

  TYPE ( f_ugo_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_fgh ), POINTER :: feval_fgh

!  ignore Fortran userdata type (not interoperable)

  TYPE ( f_nlpt_userdata_type ), POINTER :: fuserdata => NULL( )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate eval_fgh procedure pointer

  CALL C_F_PROCPOINTER( ceval_fgh, feval_fgh )

!  solve the problem using internal evaluations

  CALL f_ugo_solve_direct( fdata, fuserdata, status, x, f, g, h, wrap_eval_fgh )
  RETURN

!  wrappers

  CONTAINS

!  eval_FGH wrapper

    SUBROUTINE wrap_eval_fgh( status, x, userdata, f, g, h )     
    INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
    REAL ( KIND = wp ), INTENT( IN ) :: x
    TYPE ( f_nlpt_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h

!  call C interoperable eval_fgh

    status = feval_fgh( x, f, g, h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_fgh

  END SUBROUTINE ugo_solve_direct

!  ----------------------------------------
!  C interface to fortran ugo_solve_reverse
!  ----------------------------------------

  SUBROUTINE ugo_solve_reverse( cdata, status, eval_status,                    &
                                x, f, g, h ) BIND( C ) 
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  REAL ( KIND = wp ), INTENT( INOUT ) :: x, f, g, h
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status, eval_status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_ugo_full_data_type ), POINTER :: fdata

!  associate data pointers

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem by reverse communication

  CALL f_ugo_solve_reverse( fdata, status, eval_status, x, f, g, h )
  RETURN

  END SUBROUTINE ugo_solve_reverse

!  --------------------------------------
!  C interface to fortran ugo_information
!  --------------------------------------

  SUBROUTINE ugo_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ugo_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_ugo_full_data_type ), pointer :: fdata
  TYPE ( f_ugo_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain UGO solution information

  CALL f_ugo_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE ugo_information

!  ------------------------------------
!  C interface to fortran ugo_terminate
!  ------------------------------------

  SUBROUTINE ugo_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_UGO_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ugo_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( ugo_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_ugo_control_type ) :: fcontrol
  TYPE ( f_ugo_inform_type ) :: finform
  TYPE ( f_ugo_full_data_type ), POINTER :: fdata

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointers

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_ugo_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate fdata

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE ugo_terminate   
