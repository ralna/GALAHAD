#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.0 - 14/03/2017 AT 14:20 GMT.

!-*-*-*-  G A L A H A D _ A R C _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 14th, 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_ARC_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to ARC

      USE GALAHAD_MATLAB
      USE GALAHAD_PSLS_MATLAB_TYPES
      USE GALAHAD_GLRT_MATLAB_TYPES
      USE GALAHAD_RQS_MATLAB_TYPES
      USE GALAHAD_LMS_MATLAB_TYPES
      USE GALAHAD_SHA_MATLAB_TYPES
      USE GALAHAD_ARC_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: ARC_matlab_control_set, ARC_matlab_control_get,                &
                ARC_matlab_inform_create, ARC_matlab_inform_get

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

      TYPE, PUBLIC :: ARC_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: ARC_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, cg_iter, f_eval, g_eval, h_eval
        mwPointer :: factorization_max, factorization_status
        mwPointer :: max_entries_factors, factorization_integer
        mwPointer :: factorization_real, factorization_average
        mwPointer :: obj, norm_g
        mwPointer :: time
        TYPE ( ARC_time_pointer_type ) :: time_pointer
        TYPE ( RQS_pointer_type ) :: RQS_pointer
        TYPE ( GLRT_pointer_type ) :: GLRT_pointer
        TYPE ( PSLS_pointer_type ) :: PSLS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer_prec
        TYPE ( SHA_pointer_type ) :: SHA_pointer
      END TYPE

    CONTAINS

!-*-*-  A R C _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE ARC_matlab_control_set( ps, ARC_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to ARC

!  Arguments

!  ps - given pointer to the structure
!  ARC_control - ARC control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( ARC_control_type ) :: ARC_control

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
                                 pc, ARC_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, ARC_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, ARC_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, ARC_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, ARC_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, ARC_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, ARC_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, ARC_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, ARC_control%alive_file, len )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, ARC_control%non_monotone )
        CASE( 'model' )
          CALL MATLAB_get_value( ps, 'model',                                  &
                                 pc, ARC_control%model )
        CASE( 'norm' )
          CALL MATLAB_get_value( ps, 'norm',                                   &
                                 pc, ARC_control%norm )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, ARC_control%semi_bandwidth )
        CASE( 'lbfgs_vectors' )
          CALL MATLAB_get_value( ps, 'lbfgs_vectors',                          &
                                 pc, ARC_control%lbfgs_vectors )
        CASE( 'max_dxg' )
          CALL MATLAB_get_value( ps, 'max_dxg',                                &
                                 pc, ARC_control%max_dxg )
        CASE( 'icfs_vectors' )
          CALL MATLAB_get_value( ps, 'icfs_vectors',                           &
                                 pc, ARC_control%icfs_vectors )
        CASE( 'mi28_lsize' )
          CALL MATLAB_get_value( ps, 'mi28_lsize',                             &
                                 pc, ARC_control%mi28_lsize )
        CASE( 'mi28_rsize' )
          CALL MATLAB_get_value( ps, 'mi28_rsize',                             &
                                 pc, ARC_control%mi28_rsize )
        CASE( 'advanced_start' )
          CALL MATLAB_get_value( ps, 'advanced_start',                         &
                                 pc, ARC_control%advanced_start )
        CASE( 'stop_g_absolute' )
          CALL MATLAB_get_value( ps, 'stop_g_absolute',                        &
                                 pc, ARC_control%stop_g_absolute )
        CASE( 'stop_g_relative' )
          CALL MATLAB_get_value( ps, 'stop_g_relative',                        &
                                 pc, ARC_control%stop_g_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, ARC_control%stop_s )
        CASE( 'initial_weight' )
          CALL MATLAB_get_value( ps, 'initial_weight',                         &
                                 pc, ARC_control%initial_weight )
        CASE( 'minimum_weight' )
          CALL MATLAB_get_value( ps, 'minimum_weight',                         &
                                 pc, ARC_control%minimum_weight )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, ARC_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, ARC_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, ARC_control%eta_too_successful )
        CASE( 'weight_decrease_min' )
          CALL MATLAB_get_value( ps, 'weight_decrease_min',                    &
                                 pc, ARC_control%weight_decrease_min )
        CASE( 'weight_decrease' )
          CALL MATLAB_get_value( ps, 'weight_decrease',                        &
                                 pc, ARC_control%weight_decrease )
        CASE( 'weight_increase' )
          CALL MATLAB_get_value( ps, 'weight_increase',                        &
                                 pc, ARC_control%weight_increase )
        CASE( 'weight_increase_max' )
          CALL MATLAB_get_value( ps, 'weight_increase_max',                    &
                                 pc, ARC_control%weight_increase_max )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, ARC_control%obj_unbounded )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, ARC_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, ARC_control%clock_time_limit )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, ARC_control%hessian_available )
        CASE( 'subproblem_direct' )
          CALL MATLAB_get_value( ps, 'subproblem_direct',                      &
                                 pc, ARC_control%subproblem_direct )
        CASE( 'renormalize_weight' )
          CALL MATLAB_get_value( ps, 'renormalize_weight',                     &
                                 pc, ARC_control%renormalize_weight )
        CASE( 'quadratic_ratio_test' )
          CALL MATLAB_get_value( ps, 'quadratic_ratio_test',                     &
                                 pc, ARC_control%quadratic_ratio_test )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, ARC_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, ARC_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, ARC_control%prefix, len )
        CASE( 'RQS_control' )
          pc = mxGetField( ps, 1_mwi_, 'RQS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component RQS_control must be a structure' )
          CALL RQS_matlab_control_set( pc, ARC_control%RQS_control, len )
        CASE( 'GLRT_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLRT_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLRT_control must be a structure' )
          CALL GLRT_matlab_control_set( pc, ARC_control%GLRT_control, len )
        CASE( 'PSLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'PSLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component PSLS_control must be a structure' )
          CALL PSLS_matlab_control_set( pc, ARC_control%PSLS_control, len )
        CASE( 'LMS_control' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component LMS_control must be a structure' )
          CALL LMS_matlab_control_set( pc, ARC_control%LMS_control, len )
        CASE( 'LMS_control_prec' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control_prec' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt(' component LMS_control_prec must be a structure')
          CALL LMS_matlab_control_set( pc, ARC_control%LMS_control_prec, len )
        CASE( 'SHA_control' )
          pc = mxGetField( ps, 1_mwi_, 'SHA_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SHA_control must be a structure' )
          CALL SHA_matlab_control_set( pc, ARC_control%SHA_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine ARC_matlab_control_set

      END SUBROUTINE ARC_matlab_control_set

!-*-  A R C _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE ARC_matlab_control_get( struct, ARC_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to ARC

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  ARC_control - ARC control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ARC_control_type ) :: ARC_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 47
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'alive_unit                     ', &
         'alive_file                     ', 'non_monotone                   ', &
         'model                          ', 'norm                           ', &
         'semi_bandwidth                 ', 'lbfgs_vectors                  ', &
         'max_dxg                        ', 'icfs_vectors                   ', &
         'mi28_lsize                     ', 'mi28_rsize                     ', &
         'advanced_start                 ', 'stop_g_absolute                ', &
         'stop_g_relative                ', 'stop_s                         ', &
         'initial_weight                 ', 'minimum_weight                 ', &
         'eta_successful                 ', 'eta_very_successful            ', &
         'eta_too_successful             ', 'weight_decrease_min            ', &
         'weight_decrease                ', 'weight_increase                ', &
         'weight_increase_max            ', 'obj_unbounded                  ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'hessian_available              ', 'subproblem_direct              ', &
         'renormalize_weight             ', 'quadratic_ratio_test           ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         ', 'RQS_control                    ', &
         'GLRT_control                   ', 'PSLS_control                   ', &
         'LMS_control                    ', 'LMS_control_prec               ', &
         'SHA_control                    '                         /)

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
                                  ARC_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  ARC_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  ARC_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  ARC_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  ARC_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  ARC_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  ARC_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  ARC_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  ARC_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  ARC_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  ARC_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'model',                            &
                                  ARC_control%model )
      CALL MATLAB_fill_component( pointer, 'norm',                             &
                                  ARC_control%norm )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  ARC_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'lbfgs_vectors',                    &
                                  ARC_control%lbfgs_vectors )
      CALL MATLAB_fill_component( pointer, 'max_dxg',                          &
                                  ARC_control%max_dxg )
      CALL MATLAB_fill_component( pointer, 'icfs_vectors',                     &
                                  ARC_control%icfs_vectors )
      CALL MATLAB_fill_component( pointer, 'mi28_lsize',                       &
                                  ARC_control%mi28_lsize )
      CALL MATLAB_fill_component( pointer, 'mi28_rsize',                       &
                                  ARC_control%mi28_rsize )
      CALL MATLAB_fill_component( pointer, 'advanced_start',                   &
                                  ARC_control%advanced_start )
      CALL MATLAB_fill_component( pointer, 'stop_g_absolute',                  &
                                  ARC_control%stop_g_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_g_relative',                  &
                                  ARC_control%stop_g_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  ARC_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'initial_weight',                   &
                                  ARC_control%initial_weight )
      CALL MATLAB_fill_component( pointer, 'minimum_weight',                   &
                                  ARC_control%minimum_weight )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  ARC_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  ARC_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  ARC_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'weight_decrease_min',              &
                                  ARC_control%weight_decrease_min )
      CALL MATLAB_fill_component( pointer, 'weight_decrease',                  &
                                  ARC_control%weight_decrease )
      CALL MATLAB_fill_component( pointer, 'weight_increase',                  &
                                  ARC_control%weight_increase )
      CALL MATLAB_fill_component( pointer, 'weight_increase_max',              &
                                  ARC_control%weight_increase_max )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  ARC_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  ARC_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  ARC_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  ARC_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'subproblem_direct',                &
                                  ARC_control%subproblem_direct )
      CALL MATLAB_fill_component( pointer, 'renormalize_weight',               &
                                  ARC_control%renormalize_weight )
      CALL MATLAB_fill_component( pointer, 'quadratic_ratio_test',             &
                                  ARC_control%quadratic_ratio_test )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  ARC_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  ARC_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  ARC_control%prefix )

!  create the components of sub-structure RQS_control

      CALL RQS_matlab_control_get( pointer, ARC_control%RQS_control,           &
                                   'RQS_control' )

!  create the components of sub-structure GLRT_control

      CALL GLRT_matlab_control_get( pointer, ARC_control%GLRT_control,         &
                                  'GLRT_control' )

!  create the components of sub-structure PSLS_control

      CALL PSLS_matlab_control_get( pointer, ARC_control%PSLS_control,         &
                                  'PSLS_control' )

!  create the components of sub-structure LMS_control

      CALL LMS_matlab_control_get( pointer, ARC_control%LMS_control,           &
                                  'LMS_control' )

!  create the components of sub-structure LMS_control_prec

      CALL LMS_matlab_control_get( pointer, ARC_control%LMS_control_prec,      &
                                  'LMS_control_prec' )

!  create the components of sub-structure SHA_control

      CALL SHA_matlab_control_get( pointer, ARC_control%SHA_control,           &
                                  'SHA_control' )

      RETURN

!  End of subroutine ARC_matlab_control_get

      END SUBROUTINE ARC_matlab_control_get

!-*-  A R C _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE ARC_matlab_inform_create( struct, ARC_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold ARC_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  ARC_pointer - ARC pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ARC_pointer_type ) :: ARC_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 23
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'cg_iter              ', 'f_eval               ',                   &
           'g_eval               ', 'h_eval               ',                   &
           'factorization_status ', 'factorization_max    ',                   &
           'max_entries_factors  ', 'factorization_integer',                   &
           'factorization_real   ', 'factorization_average',                   &
           'obj                  ', 'norm_g               ',                   &
           'time                 ', 'RQS_inform           ',                   &
           'GLRT_inform          ', 'PSLS_inform          ',                   &
           'LMS_inform           ', 'LMS_inform_prec      ',                   &
           'SHA_inform           '                               /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_preprocess     ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, ARC_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        ARC_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'status', ARC_pointer%status )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
         'alloc_status', ARC_pointer%alloc_status )
      CALL MATLAB_create_char_component( ARC_pointer%pointer,                  &
        'bad_alloc', ARC_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'iter', ARC_pointer%iter )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'cg_iter', ARC_pointer%cg_iter )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'f_eval', ARC_pointer%f_eval )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'g_eval', ARC_pointer%g_eval )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'h_eval', ARC_pointer%h_eval )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'factorization_status', ARC_pointer%factorization_status )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'factorization_max', ARC_pointer%factorization_max )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'max_entries_factors', ARC_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'factorization_integer', ARC_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( ARC_pointer%pointer,               &
        'factorization_real', ARC_pointer%factorization_real )
      CALL MATLAB_create_real_component( ARC_pointer%pointer,                  &
        'factorization_average', ARC_pointer%factorization_average )
      CALL MATLAB_create_real_component( ARC_pointer%pointer,                  &
        'obj', ARC_pointer%obj )
      CALL MATLAB_create_real_component( ARC_pointer%pointer,                  &
        'norm_g', ARC_pointer%norm_g )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( ARC_pointer%pointer,                    &
        'time', ARC_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'total', ARC_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'preprocess', ARC_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'analyse', ARC_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'factorize', ARC_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'solve', ARC_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'clock_total', ARC_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'clock_preprocess', ARC_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'clock_analyse', ARC_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'clock_factorize', ARC_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( ARC_pointer%time_pointer%pointer,     &
        'clock_solve', ARC_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure RQS_inform

      CALL RQS_matlab_inform_create( ARC_pointer%pointer,                      &
                                     ARC_pointer%RQS_pointer, 'RQS_inform' )

!  Define the components of sub-structure GLRT_inform

      CALL GLRT_matlab_inform_create( ARC_pointer%pointer,                     &
                                      ARC_pointer%GLRT_pointer, 'GLRT_inform' )

!  Define the components of sub-structure PSLS_inform

      CALL PSLS_matlab_inform_create( ARC_pointer%pointer,                     &
                                      ARC_pointer%PSLS_pointer, 'PSLS_inform' )

!  Define the components of sub-structure LMS_inform

      CALL LMS_matlab_inform_create( ARC_pointer%pointer,                      &
                                     ARC_pointer%LMS_pointer, 'LMS_inform' )

!  Define the components of sub-structure LMS_inform_prec

      CALL LMS_matlab_inform_create( ARC_pointer%pointer,                      &
                                     ARC_pointer%LMS_pointer_prec,             &
                                     'LMS_inform_prec')

!  Define the components of sub-structure SHA_inform

      CALL SHA_matlab_inform_create( ARC_pointer%pointer,                      &
                                     ARC_pointer%SHA_pointer, 'SHA_inform' )

      RETURN

!  End of subroutine ARC_matlab_inform_create

      END SUBROUTINE ARC_matlab_inform_create

!-*-*-  A R C _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE ARC_matlab_inform_get( ARC_inform, ARC_pointer )

!  --------------------------------------------------------------

!  Set ARC_inform values from matlab pointers

!  Arguments

!  ARC_inform - ARC inform structure
!  ARC_pointer - ARC pointer structure

!  --------------------------------------------------------------

      TYPE ( ARC_inform_type ) :: ARC_inform
      TYPE ( ARC_pointer_type ) :: ARC_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( ARC_inform%status,                              &
                               mxGetPr( ARC_pointer%status ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%alloc_status,                        &
                               mxGetPr( ARC_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( ARC_pointer%pointer,                            &
                               'bad_alloc', ARC_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( ARC_inform%iter,                                &
                               mxGetPr( ARC_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%cg_iter,                             &
                               mxGetPr( ARC_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%f_eval,                              &
                               mxGetPr( ARC_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%g_eval,                              &
                               mxGetPr( ARC_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%h_eval,                              &
                               mxGetPr( ARC_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%factorization_status,                &
                               mxGetPr( ARC_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%factorization_max,                   &
                               mxGetPr( ARC_pointer%factorization_max ) )
      CALL galmxCopyLongToPtr( ARC_inform%max_entries_factors,                 &
                               mxGetPr( ARC_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%factorization_integer,               &
                               mxGetPr( ARC_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%factorization_real,                  &
                               mxGetPr( ARC_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%factorization_average,               &
                               mxGetPr( ARC_pointer%factorization_average ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%obj,                                 &
                               mxGetPr( ARC_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( ARC_inform%norm_g,                              &
                               mxGetPr( ARC_pointer%norm_g ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%total, wp ),              &
                           mxGetPr( ARC_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%preprocess, wp ),         &
                           mxGetPr( ARC_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%analyse, wp ),            &
                           mxGetPr( ARC_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%factorize, wp ),          &
                           mxGetPr( ARC_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%solve, wp ),              &
                           mxGetPr( ARC_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%clock_total, wp ),        &
                           mxGetPr( ARC_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%clock_preprocess, wp ),   &
                           mxGetPr( ARC_pointer%time_pointer%clock_preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%clock_analyse, wp ),      &
                           mxGetPr( ARC_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%clock_factorize, wp ),    &
                           mxGetPr( ARC_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( ARC_inform%time%clock_solve, wp ),        &
                           mxGetPr( ARC_pointer%time_pointer%clock_solve ) )

!  direct subproblem solver components

      CALL RQS_matlab_inform_get( ARC_inform%RQS_inform,                       &
                                  ARC_pointer%RQS_pointer )

!  iterative subproblem solver components

      CALL GLRT_matlab_inform_get( ARC_inform%GLRT_inform,                     &
                                   ARC_pointer%GLRT_pointer )

!  linear system solver components

      CALL PSLS_matlab_inform_get( ARC_inform%PSLS_inform,                     &
                                   ARC_pointer%PSLS_pointer )

!  limited memory solver components

      CALL LMS_matlab_inform_get( ARC_inform%LMS_inform,                       &
                                  ARC_pointer%LMS_pointer )

!  limited memory preconditioner components

      CALL LMS_matlab_inform_get( ARC_inform%LMS_inform_prec,                  &
                                  ARC_pointer%LMS_pointer_prec )

!  sparse Hessian approximation components

      CALL SHA_matlab_inform_get( ARC_inform%SHA_inform,                       &
                                  ARC_pointer%SHA_pointer )

      RETURN

!  End of subroutine ARC_matlab_inform_get

      END SUBROUTINE ARC_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ A R C _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_ARC_MATLAB_TYPES
