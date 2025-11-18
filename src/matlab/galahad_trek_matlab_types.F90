#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.4 - 2025-11-17 AT 15:50 GMT.

!-*-*-*-  G A L A H A D _ T R E K _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 12th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_TREK_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to TREK

      USE GALAHAD_MATLAB
      USE GALAHAD_TRS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_TREK_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: TREK_matlab_control_set, TREK_matlab_control_get,              &
                TREK_matlab_inform_create, TREK_matlab_inform_get

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

      TYPE, PUBLIC :: TREK_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve
        mwPointer :: clock_total, clock_assemble
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: TREK_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, n_vec, obj, x_norm, multiplier
        mwPointer :: radius, next_radius, error
        mwPointer :: time
        TYPE ( TREK_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_s_pointer
        TYPE ( TRS_pointer_type ) :: TRS_pointer
      END TYPE

    CONTAINS

!-*-  T R E K _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE TREK_matlab_control_set( ps, TREK_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to TREK

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( TREK_control_type ) :: TREK_control

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
                                 pc, TREK_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, TREK_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, TREK_control%print_level )
        CASE( 'eks_max' )
          CALL MATLAB_get_value( ps, 'eks_max',                                &
                                 pc, TREK_control%eks_max )
        CASE( 'it_max' )
          CALL MATLAB_get_value( ps, 'it_max',                                 &
                                 pc, TREK_control%it_max )
        CASE( 'f' )
          CALL MATLAB_get_value( ps, 'f',                                      &
                                 pc, TREK_control%f )
        CASE( 'reduction' )
          CALL MATLAB_get_value( ps, 'reduction',                              &
                                 pc, TREK_control%reduction )
        CASE( 'stop_residual' )
          CALL MATLAB_get_value( ps, 'stop_residual',                          &
                                 pc, TREK_control%stop_residual )
        CASE( 'reorthogonalize' )
          CALL MATLAB_get_value( ps, 'reorthogonalize',                        &
                                 pc, TREK_control%reorthogonalize )
        CASE( 's_version_52' )
          CALL MATLAB_get_value( ps, 's_version_52',                           &
                                 pc, TREK_control%s_version_52 )
        CASE( 'perturb_c' )
          CALL MATLAB_get_value( ps, 'perturb_c',                              &
                                 pc, TREK_control%perturb_c )
        CASE( 'stop_check_all_orders' )
          CALL MATLAB_get_value( ps, 'stop_check_all_orders',                  &
                                 pc, TREK_control%stop_check_all_orders )
        CASE( 'new_radius' )
          CALL MATLAB_get_value( ps, 'new_radius',                             &
                                 pc, TREK_control%new_radius )
        CASE( 'new_values' )
          CALL MATLAB_get_value( ps, 'new_values',                             &
                                 pc, TREK_control%new_values )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, TREK_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, TREK_control%deallocate_error_fatal )
        CASE( 'linear_solver' )
          CALL galmxGetCharacter( ps, 'linear_solver',                         &
                                  pc, TREK_control%linear_solver, len )
        CASE( 'linear_solver_for_s' )
          CALL galmxGetCharacter( ps, 'linear_solver_for_s',                   &
                                  pc, TREK_control%linear_solver_for_s, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, TREK_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, TREK_control%SLS_control, len )
        CASE( 'SLS_s_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_s_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_s_control must be a structure' )
          CALL SLS_matlab_control_set( pc, TREK_control%SLS_s_control, len )
        CASE( 'TRS_control' )
          pc = mxGetField( ps, 1_mwi_, 'TRS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component TRS_control must be a structure' )
          CALL TRS_matlab_control_set( pc, TREK_control%TRS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine TREK_matlab_control_set

      END SUBROUTINE TREK_matlab_control_set

!-*-  T R E K _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE TREK_matlab_control_get( struct, TREK_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to TREK

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TREK_control - TREK control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TREK_control_type ) :: TREK_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 22
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'eks_max                        ', &
         'it_max                         ', 'f                              ', &
         'reduction                      ', 'stop_residual                  ', &
         'reorthogonalize                ', 's_version_52                   ', &
         'perturb_c                      ', 'stop_check_all_orders          ', &
         'new_radius                     ', 'new_values                     ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'symmetric_linear_solver        ', 'definite_linear_solver         ', &
         'prefix                         ', 'SLS_control                    ', &
         'SLS_s_control                  ', 'TRS_control                    ' /)

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
                                  TREK_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  TREK_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  TREK_control%print_level )
      CALL MATLAB_fill_component( pointer, 'eks_max',                          &
                                  TREK_control%eks_max )
      CALL MATLAB_fill_component( pointer, 'it_max',                           &
                                  TREK_control%it_max )
      CALL MATLAB_fill_component( pointer, 'f',                                &
                                  TREK_control%f )
      CALL MATLAB_fill_component( pointer, 'reduction',                        &
                                  TREK_control%reduction )
      CALL MATLAB_fill_component( pointer, 'stop_residual',                    &
                                  TREK_control%stop_residual )
      CALL MATLAB_fill_component( pointer, 'reorthogonalize',                  &
                                  TREK_control%reorthogonalize )
      CALL MATLAB_fill_component( pointer, 's_version_52',                     &
                                  TREK_control%s_version_52 )
      CALL MATLAB_fill_component( pointer, 'perturb_c',                        &
                                  TREK_control%perturb_c )
      CALL MATLAB_fill_component( pointer, 'stop_check_all_orders',            &
                                  TREK_control%stop_check_all_orders )
      CALL MATLAB_fill_component( pointer, 'new_radius',                       &
                                  TREK_control%new_radius )
      CALL MATLAB_fill_component( pointer, 'new_values',                       &
                                  TREK_control%new_values )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  TREK_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  TREK_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'linear_solver',                    &
                                  TREK_control%linear_solver )
      CALL MATLAB_fill_component( pointer, 'linear_solver_for_s',              &
                                  TREK_control%linear_solver_for_s )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  TREK_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, TREK_control%SLS_control,          &
                                   'SLS_control' )

!  create the components of sub-structure SLS_s_control

      CALL SLS_matlab_control_get( pointer, TREK_control%SLS_s_control,        &
                                   'SLS_s_control' )

!  create the components of sub-structure TRS_control

      CALL TRS_matlab_control_get( pointer, TREK_control%TRS_control,          &
                                   'TRS_control' )

      RETURN

!  End of subroutine TREK_matlab_control_get

      END SUBROUTINE TREK_matlab_control_get

!-*-  T R E K _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -

      SUBROUTINE TREK_matlab_inform_create( struct, TREK_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold TREK_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TREK_pointer - TREK pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TREK_pointer_type ) :: TREK_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 15
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'n_vec                ', 'obj                  ',                   &
           'x_norm               ', 'multiplier           ',                   &
           'radius               ', 'next_radius          ',                   &
           'error                ', 'time                 ',                   &
           'SLS_inform           ', 'SLS_s_inform         ',                   &
           'TRS_inform           ' /)

      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'assemble             ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_assemble       ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, TREK_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        TREK_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( TREK_pointer%pointer,              &
        'status', TREK_pointer%status )
      CALL MATLAB_create_integer_component( TREK_pointer%pointer,              &
         'alloc_status', TREK_pointer%alloc_status )
      CALL MATLAB_create_char_component( TREK_pointer%pointer,                 &
        'bad_alloc', TREK_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( TREK_pointer%pointer,              &
        'iter', TREK_pointer%iter )
      CALL MATLAB_create_integer_component( TREK_pointer%pointer,              &
        'n_vec', TREK_pointer%n_vec )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'obj', TREK_pointer%obj )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'x_norm', TREK_pointer%x_norm )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'multiplier', TREK_pointer%multiplier )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'radius', TREK_pointer%radius )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'next_radius', TREK_pointer%next_radius )
      CALL MATLAB_create_real_component( TREK_pointer%pointer,                 &
        'error', TREK_pointer%error )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( TREK_pointer%pointer,                   &
        'time', TREK_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'total', TREK_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'assemble', TREK_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'analyse', TREK_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'factorize', TREK_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'solve', TREK_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'clock_total', TREK_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'clock_assemble', TREK_pointer%time_pointer%clock_assemble )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'clock_analyse', TREK_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'clock_factorize', TREK_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( TREK_pointer%time_pointer%pointer,    &
        'clock_solve', TREK_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( TREK_pointer%pointer,                     &
                                     TREK_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SLS_s_inform

      CALL SLS_matlab_inform_create( TREK_pointer%pointer,                     &
                                     TREK_pointer%SLS_s_pointer, 'SLS_s_inform')

!  Define the components of sub-structure IR_inform

      CALL TRS_matlab_inform_create( TREK_pointer%pointer,                     &
                                     TREK_pointer%TRS_pointer, 'TRS_inform' )

      RETURN

!  End of subroutine TREK_matlab_inform_create

      END SUBROUTINE TREK_matlab_inform_create

!-*-  T R E K _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE TREK_matlab_inform_get( TREK_inform, TREK_pointer )

!  --------------------------------------------------------------

!  Set TREK_inform values from matlab pointers

!  Arguments

!  TREK_inform - TREK inform structure
!  TREK_pointer - TREK pointer structure

!  --------------------------------------------------------------

      TYPE ( TREK_inform_type ) :: TREK_inform
      TYPE ( TREK_pointer_type ) :: TREK_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( TREK_inform%status,                             &
                               mxGetPr( TREK_pointer%status ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%alloc_status,                       &
                               mxGetPr( TREK_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( TREK_pointer%pointer,                           &
                               'bad_alloc', TREK_inform%bad_alloc )

      CALL MATLAB_copy_to_ptr( TREK_inform%iter,                               &
                               mxGetPr( TREK_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%n_vec,                              &
                               mxGetPr( TREK_pointer%n_vec ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%obj,                                &
                               mxGetPr( TREK_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%x_norm,                             &
                               mxGetPr( TREK_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%multiplier,                         &
                               mxGetPr( TREK_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%radius,                             &
                               mxGetPr( TREK_pointer%radius ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%next_radius,                        &
                               mxGetPr( TREK_pointer%next_radius ) )
      CALL MATLAB_copy_to_ptr( TREK_inform%error,                              &
                               mxGetPr( TREK_pointer%error ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%total, wp ),             &
                           mxGetPr( TREK_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%assemble, wp ),          &
                           mxGetPr( TREK_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%analyse, wp ),           &
                           mxGetPr( TREK_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%factorize, wp ),         &
                           mxGetPr( TREK_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%solve, wp ),             &
                           mxGetPr( TREK_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%clock_total, wp ),       &
                           mxGetPr( TREK_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%clock_assemble, wp ),    &
                           mxGetPr( TREK_pointer%time_pointer%clock_assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%clock_analyse, wp ),     &
                           mxGetPr( TREK_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%clock_factorize, wp ),   &
                           mxGetPr( TREK_pointer%time_pointer%clock_factorize ))
      CALL MATLAB_copy_to_ptr( REAL( TREK_inform%time%clock_solve, wp ),       &
                           mxGetPr( TREK_pointer%time_pointer%clock_solve ) )


!  linear system for H components

      CALL SLS_matlab_inform_get( TREK_inform%SLS_inform,                      &
                                  TREK_pointer%SLS_pointer )

!  linear system for S components

      CALL SLS_matlab_inform_get( TREK_inform%SLS_s_inform,                    &
                                  TREK_pointer%SLS_s_pointer )

!  diagonal trust-region components

      CALL TRS_matlab_inform_get( TREK_inform%TRS_inform,                      &
                                 TREK_pointer%TRS_pointer )

      RETURN

!  End of subroutine TREK_matlab_inform_get

      END SUBROUTINE TREK_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ T R E K _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_TREK_MATLAB_TYPES
