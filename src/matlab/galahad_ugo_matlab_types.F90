#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 12:50 GMT.

!-*-*-*-  G A L A H A D _ U G O _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. March 14th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_UGO_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to UGO

      USE GALAHAD_MATLAB
      USE GALAHAD_UGO_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: UGO_matlab_control_set, UGO_matlab_control_get,                &
                UGO_matlab_inform_create, UGO_matlab_inform_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: slen = 30

!--------------------------
!  Derived type definitions
!--------------------------

      TYPE, PUBLIC :: UGO_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total
        mwPointer :: clock_total
      END TYPE

      TYPE, PUBLIC :: UGO_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter
        mwPointer :: f_eval, g_eval, h_eval, dx_best
        mwPointer :: time
        TYPE ( UGO_time_pointer_type ) :: time_pointer
      END TYPE

    CONTAINS

!-*-*-  U G O _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE UGO_matlab_control_set( ps, UGO_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to UGO

!  Arguments

!  ps - given pointer to the structure
!  UGO_control - UGO control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( UGO_control_type ) :: UGO_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, UGO_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, UGO_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, UGO_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, UGO_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, UGO_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, UGO_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, UGO_control%maxit )
        CASE( 'initial_points' )
          CALL MATLAB_get_value( ps, 'initial_points',                         &
                                 pc, UGO_control%initial_points )
        CASE( 'storage_increment' )
          CALL MATLAB_get_value( ps, 'storage_increment',                      &
                                 pc, UGO_control%storage_increment )
        CASE( 'buffer' )
          CALL MATLAB_get_value( ps, 'buffer',                                 &
                                 pc, UGO_control%buffer )
        CASE( 'lipschitz_estimate_used' )
          CALL MATLAB_get_value( ps, 'lipschitz_estimate_used',                &
                                 pc, UGO_control%lipschitz_estimate_used )
        CASE( 'next_interval_selection' )
          CALL MATLAB_get_value( ps, 'next_interval_selection',                &
                                 pc, UGO_control%next_interval_selection )
        CASE( 'refine_with_newton' )
          CALL MATLAB_get_value( ps, 'refine_with_newton',                     &
                                 pc, UGO_control%refine_with_newton )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, UGO_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, UGO_control%alive_file, len )
        CASE( 'stop_length' )
          CALL MATLAB_get_value( ps, 'stop_length',                            &
                                 pc, UGO_control%stop_length )
        CASE( 'small_g_for_newton' )
          CALL MATLAB_get_value( ps, 'small_g_for_newton',                     &
                                 pc, UGO_control%small_g_for_newton )
        CASE( 'small_g' )
          CALL MATLAB_get_value( ps, 'small_g',                                &
                                 pc, UGO_control%small_g )
        CASE( 'obj_sufficient' )
          CALL MATLAB_get_value( ps, 'obj_sufficient',                         &
                                 pc, UGO_control%obj_sufficient )
        CASE( 'global_lipschitz_constant' )
          CALL MATLAB_get_value( ps, 'global_lipschitz_constant',              &
                                 pc, UGO_control%global_lipschitz_constant )
        CASE( 'reliability_parameter' )
          CALL MATLAB_get_value( ps, 'reliability_parameter',                  &
                                 pc, UGO_control%reliability_parameter )
        CASE( 'lipschitz_lower_bound' )
          CALL MATLAB_get_value( ps, 'lipschitz_lower_bound',                  &
                                 pc, UGO_control%lipschitz_lower_bound )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, UGO_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, UGO_control%clock_time_limit )
        CASE( 'second_derivative_available' )
          CALL MATLAB_get_value( ps, 'second_derivative_available',            &
                                 pc, UGO_control%second_derivative_available )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, UGO_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, UGO_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, UGO_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine UGO_matlab_control_set

      END SUBROUTINE UGO_matlab_control_set

!-*-  U G O _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE UGO_matlab_control_get( struct, UGO_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to UGO

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  UGO_control - UGO control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( UGO_control_type ) :: UGO_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 28
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'initial_points                 ', &
         'storage_increment              ', 'buffer                         ', &
         'lipschitz_estimate_used        ', 'next_interval_selection        ', &
         'refine_with_newton             ', 'alive_unit                     ', &
         'alive_file                     ', 'stop_length                    ', &
         'small_g_for_newton             ', 'small_g                        ', &
         'obj_sufficient                 ', 'global_lipschitz_constant      ', &
         'reliability_parameter          ', 'lipschitz_lower_bound          ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'second_derivative_available    ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ' /)

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
                                  UGO_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  UGO_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  UGO_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  UGO_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  UGO_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  UGO_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  UGO_control%maxit )
      CALL MATLAB_fill_component( pointer, 'initial_points',                   &
                                  UGO_control%initial_points )
      CALL MATLAB_fill_component( pointer, 'storage_increment',                &
                                  UGO_control%storage_increment )
      CALL MATLAB_fill_component( pointer, 'buffer',                           &
                                  UGO_control%buffer )
      CALL MATLAB_fill_component( pointer, 'lipschitz_estimate_used',          &
                                  UGO_control%lipschitz_estimate_used )
      CALL MATLAB_fill_component( pointer, 'next_interval_selection',          &
                                  UGO_control%next_interval_selection )
      CALL MATLAB_fill_component( pointer, 'refine_with_newton',               &
                                  UGO_control%refine_with_newton )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  UGO_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  UGO_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'stop_length',                      &
                                  UGO_control%stop_length )
      CALL MATLAB_fill_component( pointer, 'small_g_for_newton',               &
                                  UGO_control%small_g_for_newton )
      CALL MATLAB_fill_component( pointer, 'small_g',                          &
                                  UGO_control%small_g )
      CALL MATLAB_fill_component( pointer, 'obj_sufficient',                   &
                                  UGO_control%obj_sufficient )
      CALL MATLAB_fill_component( pointer, 'global_lipschitz_constant',        &
                                  UGO_control%global_lipschitz_constant )
      CALL MATLAB_fill_component( pointer, 'reliability_parameter',            &
                                  UGO_control%reliability_parameter )
      CALL MATLAB_fill_component( pointer, 'lipschitz_lower_bound',            &
                                  UGO_control%lipschitz_lower_bound )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  UGO_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  UGO_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'second_derivative_available',      &
                                  UGO_control%second_derivative_available )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  UGO_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  UGO_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  UGO_control%prefix )

      RETURN

!  End of subroutine UGO_matlab_control_get

      END SUBROUTINE UGO_matlab_control_get

!-*-  U G O _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE UGO_matlab_inform_create( struct, UGO_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold UGO_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  UGO_pointer - UGO pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( UGO_pointer_type ) :: UGO_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'f_eval               ', 'g_eval               ',                   &
           'h_eval               ', 'dx_best              ',                   &
           'time                 ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 2
      CHARACTER ( LEN = 23 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                  ', 'clock_total            ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, UGO_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        UGO_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
        'status', UGO_pointer%status )
      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
         'alloc_status', UGO_pointer%alloc_status )
      CALL MATLAB_create_char_component( UGO_pointer%pointer,                  &
        'bad_alloc', UGO_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
        'iter', UGO_pointer%iter )
      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
        'f_eval', UGO_pointer%f_eval )
      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
        'g_eval', UGO_pointer%g_eval )
      CALL MATLAB_create_integer_component( UGO_pointer%pointer,               &
        'h_eval', UGO_pointer%h_eval )
      CALL MATLAB_create_real_component( UGO_pointer%pointer,                  &
        'dx_best', UGO_pointer%dx_best )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( UGO_pointer%pointer,                    &
        'time', UGO_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( UGO_pointer%time_pointer%pointer,     &
        'total', UGO_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( UGO_pointer%time_pointer%pointer,     &
        'clock_total', UGO_pointer%time_pointer%clock_total )

      RETURN

!  End of subroutine UGO_matlab_inform_create

      END SUBROUTINE UGO_matlab_inform_create

!-*-*-  U G O _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE UGO_matlab_inform_get( UGO_inform, UGO_pointer )

!  --------------------------------------------------------------

!  Set UGO_inform values from matlab pointers

!  Arguments

!  UGO_inform - UGO inform structure
!  UGO_pointer - UGO pointer structure

!  --------------------------------------------------------------

      TYPE ( UGO_inform_type ) :: UGO_inform
      TYPE ( UGO_pointer_type ) :: UGO_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( UGO_inform%status,                              &
                               mxGetPr( UGO_pointer%status ) )
      CALL MATLAB_copy_to_ptr( UGO_inform%alloc_status,                        &
                               mxGetPr( UGO_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( UGO_pointer%pointer,                            &
                               'bad_alloc', UGO_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( UGO_inform%iter,                                &
                               mxGetPr( UGO_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( UGO_inform%f_eval,                              &
                               mxGetPr( UGO_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( UGO_inform%g_eval,                              &
                               mxGetPr( UGO_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( UGO_inform%h_eval,                              &
                               mxGetPr( UGO_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( UGO_inform%dx_best,                             &
                               mxGetPr( UGO_pointer%dx_best ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( UGO_inform%time%total, wp ),              &
             mxGetPr( UGO_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( UGO_inform%time%clock_total, wp ),        &
             mxGetPr( UGO_pointer%time_pointer%clock_total ) )

      RETURN

!  End of subroutine UGO_matlab_inform_get

      END SUBROUTINE UGO_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ U G O _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_UGO_MATLAB_TYPES
