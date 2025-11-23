#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.4 - 2025-11-22 AT 16:30 GMT.

!-*-*-*-  G A L A H A D _ N R E K _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.4. November 22nd, 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_NREK_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to NREK

      USE GALAHAD_MATLAB
      USE GALAHAD_RQS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_NREK_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: NREK_matlab_control_set, NREK_matlab_control_get,              &
                NREK_matlab_inform_create, NREK_matlab_inform_get

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

      TYPE, PUBLIC :: NREK_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve
        mwPointer :: clock_total, clock_assemble
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: NREK_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, n_vec, obj, obj_regularized, x_norm, multiplier
        mwPointer :: weight, next_weight, error
        mwPointer :: time
        TYPE ( NREK_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_s_pointer
        TYPE ( RQS_pointer_type ) :: RQS_pointer
      END TYPE

    CONTAINS

!-*-  N R E K _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE NREK_matlab_control_set( ps, NREK_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to NREK

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( NREK_control_type ) :: NREK_control

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
                                 pc, NREK_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, NREK_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, NREK_control%print_level )
        CASE( 'eks_max' )
          CALL MATLAB_get_value( ps, 'eks_max',                                &
                                 pc, NREK_control%eks_max )
        CASE( 'it_max' )
          CALL MATLAB_get_value( ps, 'it_max',                                 &
                                 pc, NREK_control%it_max )
        CASE( 'f' )
          CALL MATLAB_get_value( ps, 'f',                                      &
                                 pc, NREK_control%f )
        CASE( 'increase' )
          CALL MATLAB_get_value( ps, 'increase',                               &
                                 pc, NREK_control%increase )
        CASE( 'stop_residual' )
          CALL MATLAB_get_value( ps, 'stop_residual',                          &
                                 pc, NREK_control%stop_residual )
        CASE( 'reorthogonalize' )
          CALL MATLAB_get_value( ps, 'reorthogonalize',                        &
                                 pc, NREK_control%reorthogonalize )
        CASE( 's_version_52' )
          CALL MATLAB_get_value( ps, 's_version_52',                           &
                                 pc, NREK_control%s_version_52 )
        CASE( 'perturb_c' )
          CALL MATLAB_get_value( ps, 'perturb_c',                              &
                                 pc, NREK_control%perturb_c )
        CASE( 'stop_check_all_orders' )
          CALL MATLAB_get_value( ps, 'stop_check_all_orders',                  &
                                 pc, NREK_control%stop_check_all_orders )
        CASE( 'new_weight' )
          CALL MATLAB_get_value( ps, 'new_weight',                             &
                                 pc, NREK_control%new_weight )
        CASE( 'new_values' )
          CALL MATLAB_get_value( ps, 'new_values',                             &
                                 pc, NREK_control%new_values )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, NREK_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, NREK_control%deallocate_error_fatal )
        CASE( 'linear_solver' )
          CALL galmxGetCharacter( ps, 'linear_solver',                         &
                                  pc, NREK_control%linear_solver, len )
        CASE( 'linear_solver_for_s' )
          CALL galmxGetCharacter( ps, 'linear_solver_for_s',                   &
                                  pc, NREK_control%linear_solver_for_s, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, NREK_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, NREK_control%SLS_control, len )
        CASE( 'SLS_s_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_s_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_s_control must be a structure' )
          CALL SLS_matlab_control_set( pc, NREK_control%SLS_s_control, len )
        CASE( 'RQS_control' )
          pc = mxGetField( ps, 1_mwi_, 'RQS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component RQS_control must be a structure' )
          CALL RQS_matlab_control_set( pc, NREK_control%RQS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine NREK_matlab_control_set

      END SUBROUTINE NREK_matlab_control_set

!-*-  N R E K _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE NREK_matlab_control_get( struct, NREK_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to NREK

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NREK_control - NREK control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NREK_control_type ) :: NREK_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 22
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'eks_max                        ', &
         'it_max                         ', 'f                              ', &
         'increase                       ', 'stop_residual                  ', &
         'reorthogonalize                ', 's_version_52                   ', &
         'perturb_c                      ', 'stop_check_all_orders          ', &
         'new_weight                     ', 'new_values                     ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'linear_solver                  ', 'linear_solver_for_s            ', &
         'prefix                         ', 'SLS_control                    ', &
         'SLS_s_control                  ', 'RQS_control                    ' /)

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
                                  NREK_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  NREK_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  NREK_control%print_level )
      CALL MATLAB_fill_component( pointer, 'eks_max',                          &
                                  NREK_control%eks_max )
      CALL MATLAB_fill_component( pointer, 'it_max',                           &
                                  NREK_control%it_max )
      CALL MATLAB_fill_component( pointer, 'f',                                &
                                  NREK_control%f )
      CALL MATLAB_fill_component( pointer, 'increase',                         &
                                  NREK_control%increase )
      CALL MATLAB_fill_component( pointer, 'stop_residual',                    &
                                  NREK_control%stop_residual )
      CALL MATLAB_fill_component( pointer, 'reorthogonalize',                  &
                                  NREK_control%reorthogonalize )
      CALL MATLAB_fill_component( pointer, 's_version_52',                     &
                                  NREK_control%s_version_52 )
      CALL MATLAB_fill_component( pointer, 'perturb_c',                        &
                                  NREK_control%perturb_c )
      CALL MATLAB_fill_component( pointer, 'stop_check_all_orders',            &
                                  NREK_control%stop_check_all_orders )
      CALL MATLAB_fill_component( pointer, 'new_weight',                       &
                                  NREK_control%new_weight )
      CALL MATLAB_fill_component( pointer, 'new_values',                       &
                                  NREK_control%new_values )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  NREK_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  NREK_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'linear_solver',                    &
                                  NREK_control%linear_solver )
      CALL MATLAB_fill_component( pointer, 'linear_solver_for_s',              &
                                  NREK_control%linear_solver_for_s )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  NREK_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, NREK_control%SLS_control,          &
                                   'SLS_control' )

!  create the components of sub-structure SLS_s_control

      CALL SLS_matlab_control_get( pointer, NREK_control%SLS_s_control,        &
                                   'SLS_s_control' )

!  create the components of sub-structure RQS_control

      CALL RQS_matlab_control_get( pointer, NREK_control%RQS_control,          &
                                   'RQS_control' )

      RETURN

!  End of subroutine NREK_matlab_control_get

      END SUBROUTINE NREK_matlab_control_get

!-*-  N R E K _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -

      SUBROUTINE NREK_matlab_inform_create( struct, NREK_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold NREK_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NREK_pointer - NREK pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NREK_pointer_type ) :: NREK_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'n_vec                ', 'obj                  ',                   &
           'obj_regularized      ',                                            &
           'x_norm               ', 'multiplier           ',                   &
           'weight               ', 'next_weight          ',                   &
           'error                ', 'time                 ',                   &
           'SLS_inform           ', 'SLS_s_inform         ',                   &
           'RQS_inform           ' /)

      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'assemble             ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_assemble       ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, NREK_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        NREK_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( NREK_pointer%pointer,              &
        'status', NREK_pointer%status )
      CALL MATLAB_create_integer_component( NREK_pointer%pointer,              &
         'alloc_status', NREK_pointer%alloc_status )
      CALL MATLAB_create_char_component( NREK_pointer%pointer,                 &
        'bad_alloc', NREK_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( NREK_pointer%pointer,              &
        'iter', NREK_pointer%iter )
      CALL MATLAB_create_integer_component( NREK_pointer%pointer,              &
        'n_vec', NREK_pointer%n_vec )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'obj', NREK_pointer%obj )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'obj_regularized', NREK_pointer%obj_regularized )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'x_norm', NREK_pointer%x_norm )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'multiplier', NREK_pointer%multiplier )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'weight', NREK_pointer%weight )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'next_weight', NREK_pointer%next_weight )
      CALL MATLAB_create_real_component( NREK_pointer%pointer,                 &
        'error', NREK_pointer%error )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( NREK_pointer%pointer,                   &
        'time', NREK_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'total', NREK_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'assemble', NREK_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'analyse', NREK_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'factorize', NREK_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'solve', NREK_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'clock_total', NREK_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'clock_assemble', NREK_pointer%time_pointer%clock_assemble )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'clock_analyse', NREK_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'clock_factorize', NREK_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( NREK_pointer%time_pointer%pointer,    &
        'clock_solve', NREK_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( NREK_pointer%pointer,                     &
                                     NREK_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SLS_s_inform

      CALL SLS_matlab_inform_create( NREK_pointer%pointer,                     &
                                     NREK_pointer%SLS_s_pointer, 'SLS_s_inform')

!  Define the components of sub-structure IR_inform

      CALL RQS_matlab_inform_create( NREK_pointer%pointer,                     &
                                     NREK_pointer%RQS_pointer, 'RQS_inform' )

      RETURN

!  End of subroutine NREK_matlab_inform_create

      END SUBROUTINE NREK_matlab_inform_create

!-*-  N R E K _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE NREK_matlab_inform_get( NREK_inform, NREK_pointer )

!  --------------------------------------------------------------

!  Set NREK_inform values from matlab pointers

!  Arguments

!  NREK_inform - NREK inform structure
!  NREK_pointer - NREK pointer structure

!  --------------------------------------------------------------

      TYPE ( NREK_inform_type ) :: NREK_inform
      TYPE ( NREK_pointer_type ) :: NREK_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( NREK_inform%status,                             &
                               mxGetPr( NREK_pointer%status ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%alloc_status,                       &
                               mxGetPr( NREK_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( NREK_pointer%pointer,                           &
                               'bad_alloc', NREK_inform%bad_alloc )

      CALL MATLAB_copy_to_ptr( NREK_inform%iter,                               &
                               mxGetPr( NREK_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%n_vec,                              &
                               mxGetPr( NREK_pointer%n_vec ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%obj,                                &
                               mxGetPr( NREK_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%obj_regularized,                    &
                               mxGetPr( NREK_pointer%obj_regularized ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%x_norm,                             &
                               mxGetPr( NREK_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%multiplier,                         &
                               mxGetPr( NREK_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%weight,                             &
                               mxGetPr( NREK_pointer%weight ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%next_weight,                        &
                               mxGetPr( NREK_pointer%next_weight ) )
      CALL MATLAB_copy_to_ptr( NREK_inform%error,                              &
                               mxGetPr( NREK_pointer%error ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%total, wp ),             &
                           mxGetPr( NREK_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%assemble, wp ),          &
                           mxGetPr( NREK_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%analyse, wp ),           &
                           mxGetPr( NREK_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%factorize, wp ),         &
                           mxGetPr( NREK_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%solve, wp ),             &
                           mxGetPr( NREK_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%clock_total, wp ),       &
                           mxGetPr( NREK_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%clock_assemble, wp ),    &
                           mxGetPr( NREK_pointer%time_pointer%clock_assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%clock_analyse, wp ),     &
                           mxGetPr( NREK_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%clock_factorize, wp ),   &
                           mxGetPr( NREK_pointer%time_pointer%clock_factorize ))
      CALL MATLAB_copy_to_ptr( REAL( NREK_inform%time%clock_solve, wp ),       &
                           mxGetPr( NREK_pointer%time_pointer%clock_solve ) )


!  linear system for H components

      CALL SLS_matlab_inform_get( NREK_inform%SLS_inform,                      &
                                  NREK_pointer%SLS_pointer )

!  linear system for S components

      CALL SLS_matlab_inform_get( NREK_inform%SLS_s_inform,                    &
                                  NREK_pointer%SLS_s_pointer )

!  diagonal trust-region components

      CALL RQS_matlab_inform_get( NREK_inform%RQS_inform,                      &
                                  NREK_pointer%RQS_pointer )

      RETURN

!  End of subroutine NREK_matlab_inform_get

      END SUBROUTINE NREK_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ N R E K _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_NREK_MATLAB_TYPES
