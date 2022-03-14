#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 10:20 GMT.

!-*-*-*-  G A L A H A D _ D G O _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. March 14th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DGO_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to DGO

      USE GALAHAD_MATLAB
      USE GALAHAD_TRB_MATLAB_TYPES
      USE GALAHAD_UGO_MATLAB_TYPES
      USE GALAHAD_HASH_MATLAB_TYPES
      USE GALAHAD_DGO_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DGO_matlab_control_set, DGO_matlab_control_get,                &
                DGO_matlab_inform_create, DGO_matlab_inform_get

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

      TYPE, PUBLIC :: DGO_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, univariate_global, multivariate_local
        mwPointer :: clock_total
        mwPointer :: clock_univariate_global, clock_multivariate_local
      END TYPE

      TYPE, PUBLIC :: DGO_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: f_eval, g_eval, h_eval, obj, norm_pg
        mwPointer :: length_ratio, f_gap, why_stop
        mwPointer :: time
        TYPE ( DGO_time_pointer_type ) :: time_pointer
        TYPE ( TRB_pointer_type ) :: TRB_pointer
        TYPE ( UGO_pointer_type ) :: UGO_pointer
        TYPE ( HASH_pointer_type ) :: HASH_pointer
      END TYPE

    CONTAINS

!-*-*-  D G O _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE DGO_matlab_control_set( ps, DGO_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to DGO

!  Arguments

!  ps - given pointer to the structure
!  DGO_control - DGO control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( DGO_control_type ) :: DGO_control

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
                                 pc, DGO_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, DGO_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, DGO_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, DGO_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, DGO_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, DGO_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, DGO_control%maxit )
        CASE( 'max_evals' )
          CALL MATLAB_get_value( ps, 'max_evals',                              &
                                 pc, DGO_control%max_evals )
        CASE( 'dictionary_size' )
          CALL MATLAB_get_value( ps, 'dictionary_size',                        &
                                 pc, DGO_control%dictionary_size )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, DGO_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, DGO_control%alive_file, len )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, DGO_control%infinity )
        CASE( 'lipschitz_lower_bound' )
          CALL MATLAB_get_value( ps, 'lipschitz_lower_bound',                  &
                                 pc, DGO_control%lipschitz_lower_bound )
        CASE( 'lipschitz_reliability' )
          CALL MATLAB_get_value( ps, 'lipschitz_reliability',                  &
                                 pc, DGO_control%lipschitz_reliability )
        CASE( 'lipschitz_control' )
          CALL MATLAB_get_value( ps, 'lipschitz_control',                      &
                                 pc, DGO_control%lipschitz_control )
        CASE( 'stop_length' )
          CALL MATLAB_get_value( ps, 'stop_length',                            &
                                 pc, DGO_control%stop_length )
        CASE( 'stop_f' )
          CALL MATLAB_get_value( ps, 'stop_f',                                 &
                                 pc, DGO_control%stop_f )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, DGO_control%obj_unbounded )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, DGO_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, DGO_control%clock_time_limit )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, DGO_control%hessian_available )
        CASE( 'prune' )
          CALL MATLAB_get_value( ps, 'prune',                                  &
                                 pc, DGO_control%prune )
        CASE( 'perform_local_optimization' )
          CALL MATLAB_get_value( ps, 'perform_local_optimization',             &
                                 pc, DGO_control%perform_local_optimization )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, DGO_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, DGO_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, DGO_control%prefix, len )
        CASE( 'TRB_control' )
          pc = mxGetField( ps, 1_mwi_, 'TRB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component TRB_control must be a structure' )
          CALL TRB_matlab_control_set( pc, DGO_control%TRB_control, len )
        CASE( 'UGO_control' )
          pc = mxGetField( ps, 1_mwi_, 'UGO_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component UGO_control must be a structure' )
          CALL UGO_matlab_control_set( pc, DGO_control%UGO_control, len )
        CASE( 'HASH_control' )
          pc = mxGetField( ps, 1_mwi_, 'HASH_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component HASH_control must be a structure' )
          CALL HASH_matlab_control_set( pc, DGO_control%HASH_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine DGO_matlab_control_set

      END SUBROUTINE DGO_matlab_control_set

!-*-  D G O _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE DGO_matlab_control_get( struct, DGO_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to DGO

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DGO_control - DGO control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DGO_control_type ) :: DGO_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 29
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'max_evals                      ', &
         'dictionary_size                ', 'alive_unit                     ', &
         'alive_file                     ', 'infinity                       ', &
         'lipschitz_lower_bound          ', 'lipschitz_reliability          ', &
         'lipschitz_control              ', 'stop_length                    ', &
         'stop_f                         ', 'obj_unbounded                  ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'hessian_available              ', 'prune                          ', &
         'perform_local_optimization     ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'TRB_control                    ', 'UGO_control                    ', &
         'HASH_control                   ' /)

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
                                  DGO_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  DGO_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  DGO_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  DGO_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  DGO_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  DGO_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  DGO_control%maxit )
      CALL MATLAB_fill_component( pointer, 'max_evals',                        &
                                  DGO_control%max_evals )
      CALL MATLAB_fill_component( pointer, 'dictionary_size',                  &
                                  DGO_control%dictionary_size )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  DGO_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  DGO_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  DGO_control%infinity )
      CALL MATLAB_fill_component( pointer, 'lipschitz_lower_bound',            &
                                  DGO_control%lipschitz_lower_bound )
      CALL MATLAB_fill_component( pointer, 'lipschitz_reliability',            &
                                  DGO_control%lipschitz_reliability )
      CALL MATLAB_fill_component( pointer, 'lipschitz_control',                &
                                  DGO_control%lipschitz_control )
      CALL MATLAB_fill_component( pointer, 'stop_length',                      &
                                  DGO_control%stop_length )
      CALL MATLAB_fill_component( pointer, 'stop_f',                           &
                                  DGO_control%stop_f )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  DGO_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  DGO_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  DGO_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  DGO_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'prune',                            &
                                  DGO_control%prune )
      CALL MATLAB_fill_component( pointer, 'perform_local_optimization',       &
                                  DGO_control%perform_local_optimization )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  DGO_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  DGO_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  DGO_control%prefix )

!  create the components of sub-structure TRB_control

      CALL TRB_matlab_control_get( pointer, DGO_control%TRB_control,           &
                                   'TRB_control' )

!  create the components of sub-structure UGO_control

      CALL UGO_matlab_control_get( pointer, DGO_control%UGO_control,           &
                                  'UGO_control' )

!  create the components of sub-structure HASH_control

      CALL HASH_matlab_control_get( pointer, DGO_control%HASH_control,         &
                                  'HASH_control' )

      RETURN

!  End of subroutine DGO_matlab_control_get

      END SUBROUTINE DGO_matlab_control_get

!-*-  D G O _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE DGO_matlab_inform_create( struct, DGO_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold DGO_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DGO_pointer - DGO pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DGO_pointer_type ) :: DGO_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 15
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'f_eval               ',                   &
           'g_eval               ', 'h_eval               ',                   &
           'obj                  ', 'norm_pg              ',                   &
           'length_ratio         ', 'f_gap                ',                   &
           'why_stop             ', 'time                 ',                   &
           'TRB_inform           ', 'UGO_inform           ',                   &
           'HASH_inform          '  /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 24 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                   ', 'univariate_global       ',             &
           'multivariate_local      ', 'clock_total             ',             &
           'clock_univariate_global ', 'clock_multivariate_local' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, DGO_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        DGO_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( DGO_pointer%pointer,               &
        'status', DGO_pointer%status )
      CALL MATLAB_create_integer_component( DGO_pointer%pointer,               &
         'alloc_status', DGO_pointer%alloc_status )
      CALL MATLAB_create_char_component( DGO_pointer%pointer,                  &
        'bad_alloc', DGO_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( DGO_pointer%pointer,               &
        'f_eval', DGO_pointer%f_eval )
      CALL MATLAB_create_integer_component( DGO_pointer%pointer,               &
        'g_eval', DGO_pointer%g_eval )
      CALL MATLAB_create_integer_component( DGO_pointer%pointer,               &
        'h_eval', DGO_pointer%h_eval )
      CALL MATLAB_create_real_component( DGO_pointer%pointer,                  &
        'obj', DGO_pointer%obj )
      CALL MATLAB_create_real_component( DGO_pointer%pointer,                  &
        'norm_pg', DGO_pointer%norm_pg )
      CALL MATLAB_create_real_component( DGO_pointer%pointer,                  &
        'length_ratio', DGO_pointer%length_ratio )
      CALL MATLAB_create_real_component( DGO_pointer%pointer,                  &
        'f_gap', DGO_pointer%f_gap )
      CALL MATLAB_create_char_component( DGO_pointer%pointer,                  &
        'why_stop', DGO_pointer%why_stop )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( DGO_pointer%pointer,                    &
        'time', DGO_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'total', DGO_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'univariate_global', DGO_pointer%time_pointer%univariate_global )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'multivariate_local', DGO_pointer%time_pointer%multivariate_local )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'clock_total', DGO_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'clock_univariate_global',                                             &
        DGO_pointer%time_pointer%clock_univariate_global )
      CALL MATLAB_create_real_component( DGO_pointer%time_pointer%pointer,     &
        'clock_multivariate_local',                                            &
        DGO_pointer%time_pointer%clock_multivariate_local )

!  Define the components of sub-structure TRB_inform

      CALL TRB_matlab_inform_create( DGO_pointer%pointer,                      &
                                     DGO_pointer%TRB_pointer, 'TRB_inform' )

!  Define the components of sub-structure UGO_inform

      CALL UGO_matlab_inform_create( DGO_pointer%pointer,                      &
                                     DGO_pointer%UGO_pointer, 'UGO_inform' )

!  Define the components of sub-structure HASH_inform

      CALL HASH_matlab_inform_create( DGO_pointer%pointer,                     &
                                     DGO_pointer%HASH_pointer, 'HASH_inform' )

      RETURN

!  End of subroutine DGO_matlab_inform_create

      END SUBROUTINE DGO_matlab_inform_create

!-*-*-  D G O _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE DGO_matlab_inform_get( DGO_inform, DGO_pointer )

!  --------------------------------------------------------------

!  Set DGO_inform values from matlab pointers

!  Arguments

!  DGO_inform - DGO inform structure
!  DGO_pointer - DGO pointer structure

!  --------------------------------------------------------------

      TYPE ( DGO_inform_type ) :: DGO_inform
      TYPE ( DGO_pointer_type ) :: DGO_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( DGO_inform%status,                              &
                               mxGetPr( DGO_pointer%status ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%alloc_status,                        &
                               mxGetPr( DGO_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( DGO_pointer%pointer,                            &
                               'bad_alloc', DGO_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( DGO_inform%f_eval,                              &
                               mxGetPr( DGO_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%g_eval,                              &
                               mxGetPr( DGO_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%h_eval,                              &
                               mxGetPr( DGO_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%obj,                                 &
                               mxGetPr( DGO_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%norm_pg,                             &
                               mxGetPr( DGO_pointer%norm_pg ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%length_ratio,                        &
                               mxGetPr( DGO_pointer%length_ratio ) )
      CALL MATLAB_copy_to_ptr( DGO_inform%f_gap,                               &
                               mxGetPr( DGO_pointer%f_gap ) )
      CALL MATLAB_copy_to_ptr( DGO_pointer%pointer,                            &
                               'why_stop', DGO_inform%why_stop )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( DGO_inform%time%total, wp ),              &
             mxGetPr( DGO_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( DGO_inform%time%univariate_global, wp ),  &
             mxGetPr( DGO_pointer%time_pointer%univariate_global ) )
      CALL MATLAB_copy_to_ptr( REAL( DGO_inform%time%multivariate_local, wp ), &
             mxGetPr( DGO_pointer%time_pointer%multivariate_local ) )
      CALL MATLAB_copy_to_ptr( REAL( DGO_inform%time%clock_total, wp ),        &
             mxGetPr( DGO_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr(                                                 &
             REAL( DGO_inform%time%clock_univariate_global, wp ),              &
             mxGetPr( DGO_pointer%time_pointer%clock_univariate_global ) )
      CALL MATLAB_copy_to_ptr(                                                 &
             REAL( DGO_inform%time%clock_multivariate_local, wp ),             &
             mxGetPr( DGO_pointer%time_pointer%clock_multivariate_local ) )

!  direct subproblem solver components

      CALL TRB_matlab_inform_get( DGO_inform%TRB_inform,                       &
                                  DGO_pointer%TRB_pointer )

!  limited memory solver components

      CALL UGO_matlab_inform_get( DGO_inform%UGO_inform,                       &
                                  DGO_pointer%UGO_pointer )

!  sparse Hessian approximation components

      CALL HASH_matlab_inform_get( DGO_inform%HASH_inform,                     &
                                  DGO_pointer%HASH_pointer )

      RETURN

!  End of subroutine DGO_matlab_inform_get

      END SUBROUTINE DGO_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ D G O _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_DGO_MATLAB_TYPES
