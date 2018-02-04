#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 07/03/2011 AT 14:15 GMT.

!-*-*-*-  G A L A H A D _ R Q S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 10th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_RQS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to RQS

      USE GALAHAD_MATLAB
      USE GALAHAD_IR_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_RQS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: RQS_matlab_control_set, RQS_matlab_control_get,                &
                RQS_matlab_inform_create, RQS_matlab_inform_get

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

      TYPE, PUBLIC :: RQS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve
        mwPointer :: clock_total, clock_assemble
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: RQS_history_pointer_type
        mwPointer :: pointer
        mwPointer :: lambda, x_norm
      END TYPE

      TYPE, PUBLIC :: RQS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: factorizations, max_entries_factors, len_history
        mwPointer :: obj, obj_regularized, x_norm, multiplier, pole, hard_case
        mwPointer :: time, history
        TYPE ( RQS_time_pointer_type ) :: time_pointer
        TYPE ( RQS_history_pointer_type ) :: history_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( IR_pointer_type ) :: IR_pointer
      END TYPE

    CONTAINS

!-*-*-  R Q S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE RQS_matlab_control_set( ps, RQS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to RQS

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( RQS_control_type ) :: RQS_control

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
                                 pc, RQS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, RQS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, RQS_control%print_level )
        CASE( 'new_h' )
          CALL MATLAB_get_value( ps, 'new_h',                                  &
                                 pc, RQS_control%new_h )
        CASE( 'new_m' )
          CALL MATLAB_get_value( ps, 'new_m',                                  &
                                 pc, RQS_control%new_m )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, RQS_control%new_a )
        CASE( 'max_factorizations' )
          CALL MATLAB_get_value( ps, 'max_factorizations',                     &
                                 pc, RQS_control%max_factorizations )
        CASE( 'inverse_itmax' )
          CALL MATLAB_get_value( ps, 'inverse_itmax',                          &
                                 pc, RQS_control%inverse_itmax )
        CASE( 'taylor_max_degree' )
          CALL MATLAB_get_value( ps, 'taylor_max_degree',                      &
                                 pc, RQS_control%taylor_max_degree )
        CASE( 'initial_multiplier' )
          CALL MATLAB_get_value( ps, 'initial_multiplier',                     &
                                 pc, RQS_control%initial_multiplier )
        CASE( 'lower' )
          CALL MATLAB_get_value( ps, 'lower',                                  &
                                 pc, RQS_control%lower )
        CASE( 'upper' )
          CALL MATLAB_get_value( ps, 'upper',                                  &
                                 pc, RQS_control%upper )
        CASE( 'stop_normal' )
          CALL MATLAB_get_value( ps, 'stop_normal',                            &
                                 pc, RQS_control%stop_normal )
        CASE( 'stop_hard' )
          CALL MATLAB_get_value( ps, 'stop_hard',                              &
                                 pc, RQS_control%stop_hard )
        CASE( 'start_invit_tol' )
          CALL MATLAB_get_value( ps, 'start_invit_tol',                        &
                                 pc, RQS_control%start_invit_tol )
        CASE( 'start_invitmax_tol' )
          CALL MATLAB_get_value( ps, 'start_invitmax_tol',                     &
                                 pc, RQS_control%start_invitmax_tol )
        CASE( 'use_initial_multiplier' )
          CALL MATLAB_get_value( ps, 'use_initial_multiplier',                 &
                                 pc, RQS_control%use_initial_multiplier )
        CASE( 'initialize_approx_eigenvector' )
          CALL MATLAB_get_value( ps, 'initialize_approx_eigenvector',          &
                                 pc, RQS_control%initialize_approx_eigenvector )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, RQS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, RQS_control%deallocate_error_fatal )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, RQS_control%symmetric_linear_solver, len )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, RQS_control%definite_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, RQS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, RQS_control%SLS_control, len )
        CASE( 'IR_control' )
          pc = mxGetField( ps, 1_mwi_, 'IR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component IR_control must be a structure' )
          CALL IR_matlab_control_set( pc, RQS_control%IR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine RQS_matlab_control_set

      END SUBROUTINE RQS_matlab_control_set

!-*-  R Q S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE RQS_matlab_control_get( struct, RQS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to RQS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  RQS_control - RQS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( RQS_control_type ) :: RQS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 25
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'new_h                          ', &
         'new_m                          ', 'new_a                          ', &
         'max_factorizations             ', 'inverse_itmax                  ', &
         'taylor_max_degree              ', 'initial_multiplier             ', &
         'lower                          ', 'upper                          ', &
         'stop_normal                    ', 'stop_hard                      ', &
         'start_invit_tol                ', 'start_invitmax_tol             ', &
         'use_initial_multiplier         ', 'initialize_approx_eigenvector  ', &
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
                                  RQS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  RQS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  RQS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'new_h',                            &
                                  RQS_control%new_h )
      CALL MATLAB_fill_component( pointer, 'new_m',                            &
                                  RQS_control%new_m )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  RQS_control%new_a )
      CALL MATLAB_fill_component( pointer, 'max_factorizations',               &
                                  RQS_control%max_factorizations )
      CALL MATLAB_fill_component( pointer, 'inverse_itmax',                    &
                                  RQS_control%inverse_itmax )
      CALL MATLAB_fill_component( pointer, 'taylor_max_degree',                &
                                  RQS_control%taylor_max_degree )
      CALL MATLAB_fill_component( pointer, 'initial_multiplier',               &
                                  RQS_control%initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'lower',                            &
                                  RQS_control%lower )
      CALL MATLAB_fill_component( pointer, 'upper',                            &
                                  RQS_control%upper )
      CALL MATLAB_fill_component( pointer, 'stop_normal',                      &
                                  RQS_control%stop_normal )
      CALL MATLAB_fill_component( pointer, 'stop_hard',                        &
                                  RQS_control%stop_hard )
      CALL MATLAB_fill_component( pointer, 'start_invit_tol',                  &
                                  RQS_control%start_invit_tol )
      CALL MATLAB_fill_component( pointer, 'start_invitmax_tol',               &
                                  RQS_control%start_invitmax_tol )
      CALL MATLAB_fill_component( pointer, 'use_initial_multiplier',           &
                                  RQS_control%use_initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'initialize_approx_eigenvector',    &
                                  RQS_control%initialize_approx_eigenvector )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  RQS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  RQS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  RQS_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  RQS_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  RQS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, RQS_control%SLS_control,           &
                                   'SLS_control' )

!  create the components of sub-structure IR_control

      CALL IR_matlab_control_get( pointer, RQS_control%IR_control,             &
                                  'IR_control' )

      RETURN

!  End of subroutine RQS_matlab_control_get

      END SUBROUTINE RQS_matlab_control_get

!-*-  R Q S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE RQS_matlab_inform_create( struct, RQS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold RQS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  RQS_pointer - RQS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( RQS_pointer_type ) :: RQS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorizations       ',                   &
           'max_entries_factors  ', 'len_history          ',                   &
           'obj                  ', 'obj_regularized      ',                   &
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
        CALL MATLAB_create_substructure( struct, name, RQS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        RQS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( RQS_pointer%pointer,               &
        'status', RQS_pointer%status )
      CALL MATLAB_create_integer_component( RQS_pointer%pointer,               &
         'alloc_status', RQS_pointer%alloc_status )
      CALL MATLAB_create_char_component( RQS_pointer%pointer,                  &
        'bad_alloc', RQS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( RQS_pointer%pointer,               &
        'factorizations', RQS_pointer%factorizations )
      CALL MATLAB_create_long_component( RQS_pointer%pointer,                  &
        'max_entries_factors', RQS_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( RQS_pointer%pointer,               &
        'len_history', RQS_pointer%len_history )
      CALL MATLAB_create_logical_component( RQS_pointer%pointer,               &
        'hard_case', RQS_pointer%hard_case )
      CALL MATLAB_create_real_component( RQS_pointer%pointer,                  &
        'obj', RQS_pointer%obj )
      CALL MATLAB_create_real_component( RQS_pointer%pointer,                  &
        'obj_regularized', RQS_pointer%obj_regularized )
      CALL MATLAB_create_real_component( RQS_pointer%pointer,                  &
        'x_norm', RQS_pointer%x_norm )
      CALL MATLAB_create_real_component( RQS_pointer%pointer,                  &
        'multiplier', RQS_pointer%multiplier )
      CALL MATLAB_create_real_component( RQS_pointer%pointer,                  &
        'pole', RQS_pointer%pole )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( RQS_pointer%pointer,                    &
        'time', RQS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'total', RQS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'assemble', RQS_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'analyse', RQS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'factorize', RQS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'solve', RQS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'clock_total', RQS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'clock_assemble', RQS_pointer%time_pointer%clock_assemble )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'clock_analyse', RQS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'clock_factorize', RQS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( RQS_pointer%time_pointer%pointer,     &
        'clock_solve', RQS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure history

      CALL MATLAB_create_substructure( RQS_pointer%pointer,                    &
        'history', RQS_pointer%history_pointer%pointer, h_ninform, h_finform )
      CALL MATLAB_create_real_component(                                       &
        RQS_pointer%history_pointer%pointer,                                   &
        'lambda', history_max, RQS_pointer%history_pointer%lambda )
      CALL MATLAB_create_real_component(                                       &
        RQS_pointer%history_pointer%pointer,                                   &
        'x_norm', history_max, RQS_pointer%history_pointer%x_norm )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( RQS_pointer%pointer,                      &
                                     RQS_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure IR_inform

      CALL IR_matlab_inform_create( RQS_pointer%pointer,                       &
                                    RQS_pointer%IR_pointer, 'IR_inform' )

      RETURN

!  End of subroutine RQS_matlab_inform_create

      END SUBROUTINE RQS_matlab_inform_create

!-*-*-  R Q S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE RQS_matlab_inform_get( RQS_inform, RQS_pointer )

!  --------------------------------------------------------------

!  Set RQS_inform values from matlab pointers

!  Arguments

!  RQS_inform - RQS inform structure
!  RQS_pointer - RQS pointer structure

!  --------------------------------------------------------------

      TYPE ( RQS_inform_type ) :: RQS_inform
      TYPE ( RQS_pointer_type ) :: RQS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( RQS_inform%status,                              &
                               mxGetPr( RQS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%alloc_status,                        &
                               mxGetPr( RQS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( RQS_pointer%pointer,                            &
                               'bad_alloc', RQS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( RQS_inform%factorizations,                      &
                               mxGetPr( RQS_pointer%factorizations ) )
      CALL galmxCopyLongToPtr( RQS_inform%max_entries_factors,                 &
                               mxGetPr( RQS_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%len_history,                         &
                               mxGetPr( RQS_pointer%len_history ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%hard_case,                           &
                               mxGetPr( RQS_pointer%hard_case ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%obj,                                 &
                               mxGetPr( RQS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%obj_regularized,                     &
                               mxGetPr( RQS_pointer%obj_regularized ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%x_norm,                              &
                               mxGetPr( RQS_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%multiplier,                          &
                               mxGetPr( RQS_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( RQS_inform%pole,                                &
                               mxGetPr( RQS_pointer%pole ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%total, wp ),              &
                           mxGetPr( RQS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%assemble, wp ),           &
                           mxGetPr( RQS_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%analyse, wp ),            &
                           mxGetPr( RQS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%factorize, wp ),          &
                           mxGetPr( RQS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%solve, wp ),              &
                           mxGetPr( RQS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%clock_total, wp ),        &
                           mxGetPr( RQS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%clock_assemble, wp ),     &
                           mxGetPr( RQS_pointer%time_pointer%clock_assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%clock_analyse, wp ),      &
                           mxGetPr( RQS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%clock_factorize, wp ),    &
                           mxGetPr( RQS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( RQS_inform%time%clock_solve, wp ),        &
                           mxGetPr( RQS_pointer%time_pointer%clock_solve ) )

!  history components

      CALL MATLAB_copy_to_ptr( RQS_inform%history%lambda,                      &
                               mxGetPr( RQS_pointer%history_pointer%lambda ),  &
                               history_max )
      CALL MATLAB_copy_to_ptr( RQS_inform%history%x_norm,                      &
                               mxGetPr( RQS_pointer%history_pointer%x_norm ),  &
                               history_max )

!  linear system components

      CALL SLS_matlab_inform_get( RQS_inform%SLS_inform,                       &
                                  RQS_pointer%SLS_pointer )

!  iterative_refinement components

      CALL IR_matlab_inform_get( RQS_inform%IR_inform,                         &
                                 RQS_pointer%IR_pointer )

      RETURN

!  End of subroutine RQS_matlab_inform_get

      END SUBROUTINE RQS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ R Q S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_RQS_MATLAB_TYPES
