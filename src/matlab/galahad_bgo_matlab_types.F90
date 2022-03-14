#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 08:00 GMT.

!-*-*-*-  G A L A H A D _ B G O _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. March 14th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BGO_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BGO

      USE GALAHAD_MATLAB
      USE GALAHAD_TRB_MATLAB_TYPES
      USE GALAHAD_UGO_MATLAB_TYPES
      USE GALAHAD_LHS_MATLAB_TYPES
      USE GALAHAD_BGO_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BGO_matlab_control_set, BGO_matlab_control_get,                &
                BGO_matlab_inform_create, BGO_matlab_inform_get

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

      TYPE, PUBLIC :: BGO_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, univariate_global, multivariate_local
        mwPointer :: clock_total
        mwPointer :: clock_univariate_global, clock_multivariate_local
      END TYPE

      TYPE, PUBLIC :: BGO_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: f_eval, g_eval, h_eval, obj, norm_pg
        mwPointer :: time
        TYPE ( BGO_time_pointer_type ) :: time_pointer
        TYPE ( TRB_pointer_type ) :: TRB_pointer
        TYPE ( UGO_pointer_type ) :: UGO_pointer
        TYPE ( LHS_pointer_type ) :: LHS_pointer
      END TYPE

    CONTAINS

!-*-*-  B G O _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE BGO_matlab_control_set( ps, BGO_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BGO

!  Arguments

!  ps - given pointer to the structure
!  BGO_control - BGO control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BGO_control_type ) :: BGO_control

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
                                 pc, BGO_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BGO_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BGO_control%print_level )
        CASE( 'attempts_max' )
          CALL MATLAB_get_value( ps, 'attempts_max',                           &
                                 pc, BGO_control%attempts_max )
        CASE( 'max_evals' )
          CALL MATLAB_get_value( ps, 'max_evals',                              &
                                 pc, BGO_control%max_evals )
        CASE( 'sampling_strategy' )
          CALL MATLAB_get_value( ps, 'sampling_strategy',                      &
                                 pc, BGO_control%sampling_strategy )
        CASE( 'hypercube_discretization' )
          CALL MATLAB_get_value( ps, 'hypercube_discretization',               &
                                 pc, BGO_control%hypercube_discretization )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, BGO_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, BGO_control%alive_file, len )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BGO_control%infinity )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, BGO_control%obj_unbounded )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BGO_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, BGO_control%clock_time_limit )
        CASE( 'random_multistart' )
          CALL MATLAB_get_value( ps, 'random_multistart',                      &
                                 pc, BGO_control%random_multistart )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, BGO_control%hessian_available )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BGO_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BGO_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BGO_control%prefix, len )
        CASE( 'TRB_control' )
          pc = mxGetField( ps, 1_mwi_, 'TRB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component TRB_control must be a structure' )
          CALL TRB_matlab_control_set( pc, BGO_control%TRB_control, len )
        CASE( 'UGO_control' )
          pc = mxGetField( ps, 1_mwi_, 'UGO_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component UGO_control must be a structure' )
          CALL UGO_matlab_control_set( pc, BGO_control%UGO_control, len )
        CASE( 'LHS_control' )
          pc = mxGetField( ps, 1_mwi_, 'LHS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component LHS_control must be a structure' )
          CALL LHS_matlab_control_set( pc, BGO_control%LHS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BGO_matlab_control_set

      END SUBROUTINE BGO_matlab_control_set

!-*-  B G O _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BGO_matlab_control_get( struct, BGO_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BGO

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BGO_control - BGO control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BGO_control_type ) :: BGO_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 21
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'attempts_max                   ', &
         'max_evals                      ', 'sampling_strategy              ', &
         'hypercube_discretization       ', 'alive_unit                     ', &
         'alive_file                     ', 'infinity                       ', &
         'obj_unbounded                  ', 'cpu_time_limit                 ', &
         'clock_time_limit               ', 'random_multistart              ', &
         'hessian_available              ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'TRB_control                    ', 'UGO_control                    ', &
         'LHS_control                    ' /)

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
                                  BGO_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BGO_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BGO_control%print_level )
      CALL MATLAB_fill_component( pointer, 'attempts_max',                     &
                                  BGO_control%attempts_max )
      CALL MATLAB_fill_component( pointer, 'max_evals',                        &
                                  BGO_control%max_evals )
      CALL MATLAB_fill_component( pointer, 'sampling_strategy',                &
                                  BGO_control%sampling_strategy )
      CALL MATLAB_fill_component( pointer, 'hypercube_discretization',         &
                                  BGO_control%hypercube_discretization )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  BGO_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  BGO_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BGO_control%infinity )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  BGO_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BGO_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  BGO_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'random_multistart',                &
                                  BGO_control%random_multistart )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  BGO_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BGO_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BGO_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BGO_control%prefix )

!  create the components of sub-structure TRB_control

      CALL TRB_matlab_control_get( pointer, BGO_control%TRB_control,           &
                                   'TRB_control' )

!  create the components of sub-structure UGO_control

      CALL UGO_matlab_control_get( pointer, BGO_control%UGO_control,           &
                                  'UGO_control' )

!  create the components of sub-structure LHS_control

      CALL LHS_matlab_control_get( pointer, BGO_control%LHS_control,           &
                                  'LHS_control' )

      RETURN

!  End of subroutine BGO_matlab_control_get

      END SUBROUTINE BGO_matlab_control_get

!-*-  B G O _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE BGO_matlab_inform_create( struct, BGO_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BGO_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BGO_pointer - BGO pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BGO_pointer_type ) :: BGO_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 12
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'f_eval               ',                   &
           'g_eval               ', 'h_eval               ',                   &
           'obj                  ', 'norm_pg              ',                   &
           'time                 ', 'TRB_inform           ',                   &
           'UGO_inform           ', 'LHS_inform           '  /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 24 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                   ', 'univariate_global       ',             &
           'multivariate_local      ', 'clock_total             ',             &
           'clock_univariate_global ', 'clock_multivariate_local' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, BGO_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BGO_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BGO_pointer%pointer,               &
        'status', BGO_pointer%status )
      CALL MATLAB_create_integer_component( BGO_pointer%pointer,               &
         'alloc_status', BGO_pointer%alloc_status )
      CALL MATLAB_create_char_component( BGO_pointer%pointer,                  &
        'bad_alloc', BGO_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BGO_pointer%pointer,               &
        'f_eval', BGO_pointer%f_eval )
      CALL MATLAB_create_integer_component( BGO_pointer%pointer,               &
        'g_eval', BGO_pointer%g_eval )
      CALL MATLAB_create_integer_component( BGO_pointer%pointer,               &
        'h_eval', BGO_pointer%h_eval )
      CALL MATLAB_create_real_component( BGO_pointer%pointer,                  &
        'obj', BGO_pointer%obj )
      CALL MATLAB_create_real_component( BGO_pointer%pointer,                  &
        'norm_pg', BGO_pointer%norm_pg )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BGO_pointer%pointer,                    &
        'time', BGO_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'total', BGO_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'univariate_global', BGO_pointer%time_pointer%univariate_global )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'multivariate_local', BGO_pointer%time_pointer%multivariate_local )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'clock_total', BGO_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'clock_univariate_global',                                             &
        BGO_pointer%time_pointer%clock_univariate_global )
      CALL MATLAB_create_real_component( BGO_pointer%time_pointer%pointer,     &
        'clock_multivariate_local',                                            &
        BGO_pointer%time_pointer%clock_multivariate_local )

!  Define the components of sub-structure TRB_inform

      CALL TRB_matlab_inform_create( BGO_pointer%pointer,                      &
                                     BGO_pointer%TRB_pointer, 'TRB_inform' )

!  Define the components of sub-structure UGO_inform

      CALL UGO_matlab_inform_create( BGO_pointer%pointer,                      &
                                     BGO_pointer%UGO_pointer, 'UGO_inform' )

!  Define the components of sub-structure LHS_inform

      CALL LHS_matlab_inform_create( BGO_pointer%pointer,                      &
                                     BGO_pointer%LHS_pointer, 'LHS_inform' )

      RETURN

!  End of subroutine BGO_matlab_inform_create

      END SUBROUTINE BGO_matlab_inform_create

!-*-*-  B G O _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE BGO_matlab_inform_get( BGO_inform, BGO_pointer )

!  --------------------------------------------------------------

!  Set BGO_inform values from matlab pointers

!  Arguments

!  BGO_inform - BGO inform structure
!  BGO_pointer - BGO pointer structure

!  --------------------------------------------------------------

      TYPE ( BGO_inform_type ) :: BGO_inform
      TYPE ( BGO_pointer_type ) :: BGO_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BGO_inform%status,                              &
                               mxGetPr( BGO_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BGO_inform%alloc_status,                        &
                               mxGetPr( BGO_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BGO_pointer%pointer,                            &
                               'bad_alloc', BGO_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BGO_inform%f_eval,                              &
                               mxGetPr( BGO_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( BGO_inform%g_eval,                              &
                               mxGetPr( BGO_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( BGO_inform%h_eval,                              &
                               mxGetPr( BGO_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( BGO_inform%obj,                                 &
                               mxGetPr( BGO_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( BGO_inform%norm_pg,                             &
                               mxGetPr( BGO_pointer%norm_pg ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( BGO_inform%time%total, wp ),              &
             mxGetPr( BGO_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BGO_inform%time%univariate_global, wp ),  &
             mxGetPr( BGO_pointer%time_pointer%univariate_global ) )
      CALL MATLAB_copy_to_ptr( REAL( BGO_inform%time%multivariate_local, wp ), &
             mxGetPr( BGO_pointer%time_pointer%multivariate_local ) )
      CALL MATLAB_copy_to_ptr( REAL( BGO_inform%time%clock_total, wp ),        &
             mxGetPr( BGO_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr(                                                 &
             REAL( BGO_inform%time%clock_univariate_global, wp ),              &
             mxGetPr( BGO_pointer%time_pointer%clock_univariate_global ) )
      CALL MATLAB_copy_to_ptr(                                                 &
             REAL( BGO_inform%time%clock_multivariate_local, wp ),             &
             mxGetPr( BGO_pointer%time_pointer%clock_multivariate_local ) )

!  direct subproblem solver components

      CALL TRB_matlab_inform_get( BGO_inform%TRB_inform,                       &
                                  BGO_pointer%TRB_pointer )

!  limited memory solver components

      CALL UGO_matlab_inform_get( BGO_inform%UGO_inform,                       &
                                  BGO_pointer%UGO_pointer )

!  sparse Hessian approximation components

      CALL LHS_matlab_inform_get( BGO_inform%LHS_inform,                       &
                                  BGO_pointer%LHS_pointer )

      RETURN

!  End of subroutine BGO_matlab_inform_get

      END SUBROUTINE BGO_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B G O _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_BGO_MATLAB_TYPES
