#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.0 - 24/03/2018 AT 14:40 GMT.

!-*-*-*-  G A L A H A D _ D P S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. Match 24th, 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DPS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to DPS

      USE GALAHAD_MATLAB
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_DPS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DPS_matlab_control_set, DPS_matlab_control_get,                &
                DPS_matlab_inform_create, DPS_matlab_inform_get

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

      TYPE, PUBLIC :: DPS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
        mwPointer :: clock_total, clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: DPS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, mod_1by1, mod_2by2
        mwPointer :: obj,  obj_regularized, x_norm, multiplier, pole, hard_case
        mwPointer :: time
        TYPE ( DPS_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
      END TYPE

    CONTAINS

!-*-*-  T R S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE DPS_matlab_control_set( ps, DPS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to DPS

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( DPS_control_type ) :: DPS_control

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
                                 pc, DPS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, DPS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, DPS_control%print_level )
        CASE( 'new_h' )
          CALL MATLAB_get_value( ps, 'new_h',                                  &
                                 pc, DPS_control%new_h )
                                 pc, DPS_control%inverse_itmax )
        CASE( 'taylor_max_degree' )
          CALL MATLAB_get_value( ps, 'taylor_max_degree',                      &
                                 pc, DPS_control%taylor_max_degree )
        CASE( 'eigen_min' )
          CALL MATLAB_get_value( ps, 'eigen_min',                              &
                                 pc, DPS_control%eigen_min )
        CASE( 'lower' )
          CALL MATLAB_get_value( ps, 'lower',                                  &
                                 pc, DPS_control%lower )
        CASE( 'upper' )
          CALL MATLAB_get_value( ps, 'upper',                                  &
                                 pc, DPS_control%upper )
        CASE( 'stop_normal' )
          CALL MATLAB_get_value( ps, 'stop_normal',                            &
                                 pc, DPS_control%stop_normal )
        CASE( 'stop_absolute_normal' )
          CALL MATLAB_get_value( ps, 'stop_absolute_normal',                   &
                                 pc, DPS_control%stop_absolute_normal )
        CASE( 'goldfarb' )
          CALL MATLAB_get_value( ps, 'goldfarb',                               &
                                 pc, DPS_control%force_Newton )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, DPS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, DPS_control%deallocate_error_fatal )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, DPS_control%symmetric_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, DPS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, DPS_control%SLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine DPS_matlab_control_set

      END SUBROUTINE DPS_matlab_control_set

!-*-  T R S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE DPS_matlab_control_get( struct, DPS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to DPS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DPS_control - DPS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DPS_control_type ) :: DPS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'new_h                          ', &
         'taylor_max_degree              ', 'eigen_min                      ', &
         'lower                          ', 'upper                          ', &
         'stop_normal                    ', 'stop_absolute_normal           ', &
         'goldfarb                       ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'symmetric_linear_solver        ', &
         'prefix                         ', 'SLS_control                    ' /)

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
                                  DPS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  DPS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  DPS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'new_h',                            &
                                  DPS_control%new_h )
      CALL MATLAB_fill_component( pointer, 'taylor_max_degree',                &
                                  DPS_control%taylor_max_degree )
      CALL MATLAB_fill_component( pointer, 'eigen_min',                        &
                                  DPS_control%eigen_min )
      CALL MATLAB_fill_component( pointer, 'lower',                            &
                                  DPS_control%lower )
      CALL MATLAB_fill_component( pointer, 'upper',                            &
                                  DPS_control%upper )
      CALL MATLAB_fill_component( pointer, 'stop_normal',                      &
                                  DPS_control%stop_normal )
      CALL MATLAB_fill_component( pointer, 'stop_absolute_normal',             &
                                  DPS_control%stop_absolute_normal )
      CALL MATLAB_fill_component( pointer, 'goldfarb',                         &
                                  DPS_control%goldfarb )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  DPS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  DPS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  DPS_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  DPS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, DPS_control%SLS_control,           &
                                   'SLS_control' )

      RETURN

!  End of subroutine DPS_matlab_control_get

      END SUBROUTINE DPS_matlab_control_get

!-*-  T R S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE DPS_matlab_inform_create( struct, DPS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold DPS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DPS_pointer - DPS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DPS_pointer_type ) :: DPS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 13
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'mod_1by1             ',                   &
           'mod_2by2             ', 'obj                  ',                   &
           'obj_regularized      ', 'x_norm               ',                   &
           'multiplier           ', 'pole                 ',                   &
           'hard_case            ', 'time                 ',                   &
           'SLS_inform           ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'analyse              ',                   &
           'factorize            ', 'solve                ',                   &
           'clock_total          ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, DPS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        DPS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( DPS_pointer%pointer,               &
        'status', DPS_pointer%status )
      CALL MATLAB_create_integer_component( DPS_pointer%pointer,               &
         'alloc_status', DPS_pointer%alloc_status )
      CALL MATLAB_create_char_component( DPS_pointer%pointer,                  &
        'bad_alloc', DPS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( DPS_pointer%pointer,               &
         'mod_1by1', DPS_pointer%mod_1by1 )
      CALL MATLAB_create_integer_component( DPS_pointer%pointer,               &
         'mod_2by2', DPS_pointer%mod_2by2 )
      CALL MATLAB_create_logical_component( DPS_pointer%pointer,               &
        'hard_case', DPS_pointer%hard_case )
      CALL MATLAB_create_real_component( DPS_pointer%pointer,                  &
        'obj', DPS_pointer%obj )
      CALL MATLAB_create_real_component( DPS_pointer%pointer,                  &
        'obj_regularized', DPS_pointer%obj_regularized )
      CALL MATLAB_create_real_component( DPS_pointer%pointer,                  &
        'x_norm', DPS_pointer%x_norm )
      CALL MATLAB_create_real_component( DPS_pointer%pointer,                  &
        'multiplier', DPS_pointer%multiplier )
      CALL MATLAB_create_real_component( DPS_pointer%pointer,                  &
        'pole', DPS_pointer%pole )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( DPS_pointer%pointer,                    &
        'time', DPS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'total', DPS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'analyse', DPS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'factorize', DPS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'solve', DPS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'clock_total', DPS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'clock_analyse', DPS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'clock_factorize', DPS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( DPS_pointer%time_pointer%pointer,     &
        'clock_solve', DPS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( DPS_pointer%pointer,                      &
                                     DPS_pointer%SLS_pointer, 'SLS_inform' )

      RETURN

!  End of subroutine DPS_matlab_inform_create

      END SUBROUTINE DPS_matlab_inform_create

!-*-*-  T R S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE DPS_matlab_inform_get( DPS_inform, DPS_pointer )

!  --------------------------------------------------------------

!  Set DPS_inform values from matlab pointers

!  Arguments

!  DPS_inform - DPS inform structure
!  DPS_pointer - DPS pointer structure

!  --------------------------------------------------------------

      TYPE ( DPS_inform_type ) :: DPS_inform
      TYPE ( DPS_pointer_type ) :: DPS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( DPS_inform%status,                              &
                               mxGetPr( DPS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%alloc_status,                        &
                               mxGetPr( DPS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( DPS_pointer%pointer,                            &
                               'bad_alloc', DPS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( DPS_inform%mod_1by1,                            &
                               mxGetPr( DPS_pointer%mod_1by1 ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%mod_1by1,                            &
                               mxGetPr( DPS_pointer%mod_1by1 ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%hard_case,                           &
                               mxGetPr( DPS_pointer%hard_case ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%obj,                                 &
                               mxGetPr( DPS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%obj_regularized,                     &
                               mxGetPr( DPS_pointer%obj_regularized ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%x_norm,                              &
                               mxGetPr( DPS_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%multiplier,                          &
                               mxGetPr( DPS_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( DPS_inform%pole,                                &
                               mxGetPr( DPS_pointer%pole ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%total, wp ),              &
                           mxGetPr( DPS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%analyse, wp ),            &
                           mxGetPr( DPS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%factorize, wp ),          &
                           mxGetPr( DPS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%solve, wp ),              &
                           mxGetPr( DPS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%clock_total, wp ),        &
                           mxGetPr( DPS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%clock_analyse, wp ),      &
                           mxGetPr( DPS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%clock_factorize, wp ),    &
                           mxGetPr( DPS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( DPS_inform%time%clock_solve, wp ),        &
                           mxGetPr( DPS_pointer%time_pointer%clock_solve ) )

!  linear system components

      CALL SLS_matlab_inform_get( DPS_inform%SLS_inform,                       &
                                  DPS_pointer%SLS_pointer )

      RETURN

!  End of subroutine DPS_matlab_inform_get

      END SUBROUTINE DPS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ D P S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_DPS_MATLAB_TYPES
