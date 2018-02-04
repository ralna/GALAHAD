#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 07/03/2011 AT 14:00 GMT.

!-*-*-*-  G A L A H A D _ T R S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 12th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_TRS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to TRS

      USE GALAHAD_MATLAB
      USE GALAHAD_IR_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_TRS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: TRS_matlab_control_set, TRS_matlab_control_get,                &
                TRS_matlab_inform_create, TRS_matlab_inform_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: slen = 30
      INTEGER, PARAMETER :: history_max = 100

!--------------------------
!  Derived type definitions
!--------------------------

      TYPE, PUBLIC :: TRS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve
        mwPointer :: clock_total, clock_assemble
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: TRS_history_pointer_type
        mwPointer :: pointer
        mwPointer :: lambda, x_norm
      END TYPE

      TYPE, PUBLIC :: TRS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: factorizations, max_entries_factors, len_history
        mwPointer :: obj, x_norm, multiplier, pole, hard_case
        mwPointer :: time, history
        TYPE ( TRS_time_pointer_type ) :: time_pointer
        TYPE ( TRS_history_pointer_type ) :: history_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( IR_pointer_type ) :: IR_pointer
      END TYPE

    CONTAINS

!-*-*-  T R S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE TRS_matlab_control_set( ps, TRS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to TRS

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( TRS_control_type ) :: TRS_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc, mxGetField
      mwSize :: mxGetNumberOfFields
      LOGICAL :: mxIsStruct
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, TRS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, TRS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, TRS_control%print_level )
        CASE( 'dense_factorization' )
          CALL MATLAB_get_value( ps, 'dense_factorization',                    &
                                 pc, TRS_control%dense_factorization )
        CASE( 'new_h' )
          CALL MATLAB_get_value( ps, 'new_h',                                  &
                                 pc, TRS_control%new_h )
        CASE( 'new_m' )
          CALL MATLAB_get_value( ps, 'new_m',                                  &
                                 pc, TRS_control%new_m )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, TRS_control%new_a )
        CASE( 'max_factorizations' )
          CALL MATLAB_get_value( ps, 'max_factorizations',                     &
                                 pc, TRS_control%max_factorizations )
        CASE( 'inverse_itmax' )
          CALL MATLAB_get_value( ps, 'inverse_itmax',                          &
                                 pc, TRS_control%inverse_itmax )
        CASE( 'taylor_max_degree' )
          CALL MATLAB_get_value( ps, 'taylor_max_degree',                      &
                                 pc, TRS_control%taylor_max_degree )
        CASE( 'initial_multiplier' )
          CALL MATLAB_get_value( ps, 'initial_multiplier',                     &
                                 pc, TRS_control%initial_multiplier )
        CASE( 'lower' )
          CALL MATLAB_get_value( ps, 'lower',                                  &
                                 pc, TRS_control%lower )
        CASE( 'upper' )
          CALL MATLAB_get_value( ps, 'upper',                                  &
                                 pc, TRS_control%upper )
        CASE( 'stop_normal' )
          CALL MATLAB_get_value( ps, 'stop_normal',                            &
                                 pc, TRS_control%stop_normal )
        CASE( 'stop_absolute_normal' )
          CALL MATLAB_get_value( ps, 'stop_absolute_normal',                   &
                                 pc, TRS_control%stop_absolute_normal )
        CASE( 'stop_hard' )
          CALL MATLAB_get_value( ps, 'stop_hard',                              &
                                 pc, TRS_control%stop_hard )
        CASE( 'start_invit_tol' )
          CALL MATLAB_get_value( ps, 'start_invit_tol',                        &
                                 pc, TRS_control%start_invit_tol )
        CASE( 'start_invitmax_tol' )
          CALL MATLAB_get_value( ps, 'start_invitmax_tol',                     &
                                 pc, TRS_control%start_invitmax_tol )
        CASE( 'use_initial_multiplier' )
          CALL MATLAB_get_value( ps, 'use_initial_multiplier',                 &
                                 pc, TRS_control%use_initial_multiplier )
        CASE( 'equality_problem' )
          CALL MATLAB_get_value( ps, 'equality_problem',                       &
                                 pc, TRS_control%equality_problem )
        CASE( 'initialize_approx_eigenvector' )
          CALL MATLAB_get_value( ps, 'initialize_approx_eigenvector',          &
                                 pc, TRS_control%initialize_approx_eigenvector )
        CASE( 'force_Newton' )
          CALL MATLAB_get_value( ps, 'force_Newton',                           &
                                 pc, TRS_control%force_Newton )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, TRS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, TRS_control%deallocate_error_fatal )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, TRS_control%symmetric_linear_solver, len )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, TRS_control%definite_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, TRS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, TRS_control%SLS_control, len )
        CASE( 'IR_control' )
          pc = mxGetField( ps, 1_mwi_, 'IR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component IR_control must be a structure' )
          CALL IR_matlab_control_set( pc, TRS_control%IR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine TRS_matlab_control_set

      END SUBROUTINE TRS_matlab_control_set

!-*-  T R S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE TRS_matlab_control_get( struct, TRS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to TRS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRS_control - TRS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRS_control_type ) :: TRS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 29
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'dense_factorization            ', &
         'new_h                          ',                                    &
         'new_m                          ', 'new_a                          ', &
         'max_factorizations             ', 'inverse_itmax                  ', &
         'taylor_max_degree              ', 'initial_multiplier             ', &
         'lower                          ', 'upper                          ', &
         'stop_normal                    ', 'stop_absolute_normal           ', &
         'stop_hard                      ',                                    &
         'start_invit_tol                ', 'start_invitmax_tol             ', &
         'use_initial_multiplier         ', 'equality_problem               ', &
         'initialize_approx_eigenvector  ', 'force_Newton                   ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'symmetric_linear_solver        ', 'definite_linear_solver         ', &
         'prefix                         ', 'SLS_control                    ', &
         'IR_control                     '                      /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, pointer,                &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        pointer = struct
      END IF

!  create the components and get the values

      CALL MATLAB_fill_component( pointer, 'error',                            &
                                  TRS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  TRS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  TRS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'dense_factorization',              &
                                  TRS_control%dense_factorization )
      CALL MATLAB_fill_component( pointer, 'new_h',                            &
                                  TRS_control%new_h )
      CALL MATLAB_fill_component( pointer, 'new_m',                            &
                                  TRS_control%new_m )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  TRS_control%new_a )
      CALL MATLAB_fill_component( pointer, 'max_factorizations',               &
                                  TRS_control%max_factorizations )
      CALL MATLAB_fill_component( pointer, 'inverse_itmax',                    &
                                  TRS_control%inverse_itmax )
      CALL MATLAB_fill_component( pointer, 'taylor_max_degree',                &
                                  TRS_control%taylor_max_degree )
      CALL MATLAB_fill_component( pointer, 'initial_multiplier',               &
                                  TRS_control%initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'lower',                            &
                                  TRS_control%lower )
      CALL MATLAB_fill_component( pointer, 'upper',                            &
                                  TRS_control%upper )
      CALL MATLAB_fill_component( pointer, 'stop_normal',                      &
                                  TRS_control%stop_normal )
      CALL MATLAB_fill_component( pointer, 'stop_absolute_normal',             &
                                  TRS_control%stop_absolute_normal )
      CALL MATLAB_fill_component( pointer, 'stop_hard',                        &
                                  TRS_control%stop_hard )
      CALL MATLAB_fill_component( pointer, 'start_invit_tol',                  &
                                  TRS_control%start_invit_tol )
      CALL MATLAB_fill_component( pointer, 'start_invitmax_tol',               &
                                  TRS_control%start_invitmax_tol )
      CALL MATLAB_fill_component( pointer, 'use_initial_multiplier',           &
                                  TRS_control%use_initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'equality_problem',                 &
                                  TRS_control%equality_problem )
      CALL MATLAB_fill_component( pointer, 'initialize_approx_eigenvector',    &
                                  TRS_control%initialize_approx_eigenvector )
      CALL MATLAB_fill_component( pointer, 'force_Newton',                     &
                                  TRS_control%force_Newton )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  TRS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  TRS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  TRS_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  TRS_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  TRS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, TRS_control%SLS_control,           &
                                   'SLS_control' )

!  create the components of sub-structure IR_control

      CALL IR_matlab_control_get( pointer, TRS_control%IR_control,             &
                                  'IR_control' )

      RETURN

!  End of subroutine TRS_matlab_control_get

      END SUBROUTINE TRS_matlab_control_get

!-*-  T R S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE TRS_matlab_inform_create( struct, TRS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold TRS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRS_pointer - TRS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRS_pointer_type ) :: TRS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 15
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorizations       ',                   &
           'max_entries_factors  ',                                            &
           'len_history          ', 'obj                  ',                   &
           'x_norm               ', 'multiplier           ',                   &
           'pole                 ', 'hard_case            ',                   &
           'time                 ', 'history              ',                   &
           'SLS_inform           ', 'IR_inform            ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'assemble             ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_assemble       ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)
      INTEGER * 4, PARAMETER :: h_ninform = 2
      CHARACTER ( LEN = 21 ), PARAMETER :: h_finform( h_ninform ) = (/         &
           'lambda               ', 'x_norm               ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, TRS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        TRS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( TRS_pointer%pointer,               &
        'status', TRS_pointer%status )
      CALL MATLAB_create_integer_component( TRS_pointer%pointer,               &
         'alloc_status', TRS_pointer%alloc_status )
      CALL MATLAB_create_char_component( TRS_pointer%pointer,                  &
        'bad_alloc', TRS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( TRS_pointer%pointer,               &
        'factorizations', TRS_pointer%factorizations )
      CALL MATLAB_create_long_component( TRS_pointer%pointer,                  &
        'max_entries_factors', TRS_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( TRS_pointer%pointer,               &
        'len_history', TRS_pointer%len_history )
      CALL MATLAB_create_logical_component( TRS_pointer%pointer,               &
        'hard_case', TRS_pointer%hard_case )
      CALL MATLAB_create_real_component( TRS_pointer%pointer,                  &
        'obj', TRS_pointer%obj )
      CALL MATLAB_create_real_component( TRS_pointer%pointer,                  &
        'x_norm', TRS_pointer%x_norm )
      CALL MATLAB_create_real_component( TRS_pointer%pointer,                  &
        'multiplier', TRS_pointer%multiplier )
      CALL MATLAB_create_real_component( TRS_pointer%pointer,                  &
        'pole', TRS_pointer%pole )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( TRS_pointer%pointer,                    &
        'time', TRS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'total', TRS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'assemble', TRS_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'analyse', TRS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'factorize', TRS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'solve', TRS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'clock_total', TRS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'clock_assemble', TRS_pointer%time_pointer%clock_assemble )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'clock_analyse', TRS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'clock_factorize', TRS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( TRS_pointer%time_pointer%pointer,     &
        'clock_solve', TRS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure history

      CALL MATLAB_create_substructure( TRS_pointer%pointer,                    &
        'history', TRS_pointer%history_pointer%pointer, h_ninform, h_finform )
      CALL MATLAB_create_real_component(                                       &
        TRS_pointer%history_pointer%pointer,                                   &
        'lambda', history_max, TRS_pointer%history_pointer%lambda )
      CALL MATLAB_create_real_component(                                       &
        TRS_pointer%history_pointer%pointer,                                   &
        'x_norm', history_max, TRS_pointer%history_pointer%x_norm )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( TRS_pointer%pointer,                      &
                                     TRS_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure IR_inform

      CALL IR_matlab_inform_create( TRS_pointer%pointer,                       &
                                    TRS_pointer%IR_pointer, 'IR_inform' )

      RETURN

!  End of subroutine TRS_matlab_inform_create

      END SUBROUTINE TRS_matlab_inform_create

!-*-*-  T R S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE TRS_matlab_inform_get( TRS_inform, TRS_pointer )

!  --------------------------------------------------------------

!  Set TRS_inform values from matlab pointers

!  Arguments

!  TRS_inform - TRS inform structure
!  TRS_pointer - TRS pointer structure

!  --------------------------------------------------------------

      TYPE ( TRS_inform_type ) :: TRS_inform
      TYPE ( TRS_pointer_type ) :: TRS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( TRS_inform%status,                              &
                               mxGetPr( TRS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%alloc_status,                        &
                               mxGetPr( TRS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( TRS_pointer%pointer,                            &
                               'bad_alloc', TRS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( TRS_inform%factorizations,                      &
                               mxGetPr( TRS_pointer%factorizations ) )
      CALL galmxCopyLongToPtr( TRS_inform%max_entries_factors,                 &
                               mxGetPr( TRS_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%len_history,                         &
                               mxGetPr( TRS_pointer%len_history ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%hard_case,                           &
                               mxGetPr( TRS_pointer%hard_case ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%obj,                                 &
                               mxGetPr( TRS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%x_norm,                              &
                               mxGetPr( TRS_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%multiplier,                          &
                               mxGetPr( TRS_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( TRS_inform%pole,                                &
                               mxGetPr( TRS_pointer%pole ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%total, wp ),              &
                           mxGetPr( TRS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%assemble, wp ),           &
                           mxGetPr( TRS_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%analyse, wp ),            &
                           mxGetPr( TRS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%factorize, wp ),          &
                           mxGetPr( TRS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%solve, wp ),              &
                           mxGetPr( TRS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%clock_total, wp ),        &
                           mxGetPr( TRS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%clock_assemble, wp ),     &
                           mxGetPr( TRS_pointer%time_pointer%clock_assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%clock_analyse, wp ),      &
                           mxGetPr( TRS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%clock_factorize, wp ),    &
                           mxGetPr( TRS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRS_inform%time%clock_solve, wp ),        &
                           mxGetPr( TRS_pointer%time_pointer%clock_solve ) )

!  history components

      CALL MATLAB_copy_to_ptr( TRS_inform%history%lambda,                      &
                               mxGetPr( TRS_pointer%history_pointer%lambda ),  &
                               history_max )
      CALL MATLAB_copy_to_ptr( TRS_inform%history%x_norm,                      &
                               mxGetPr( TRS_pointer%history_pointer%x_norm ),  &
                               history_max )

!  linear system components

      CALL SLS_matlab_inform_get( TRS_inform%SLS_inform,                       &
                                  TRS_pointer%SLS_pointer )

!  iterative_refinement components

      CALL IR_matlab_inform_get( TRS_inform%IR_inform,                         &
                                 TRS_pointer%IR_pointer )

      RETURN

!  End of subroutine TRS_matlab_inform_get

      END SUBROUTINE TRS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ T R S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_TRS_MATLAB_TYPES
