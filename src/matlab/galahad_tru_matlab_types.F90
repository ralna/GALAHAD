#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.0 - 02/03/2017 AT 10:00 GMT.

!-*-*-*-  G A L A H A D _ T R U _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 2nd, 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_TRU_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to TRU

      USE GALAHAD_MATLAB
      USE GALAHAD_PSLS_MATLAB_TYPES
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_TRS_MATLAB_TYPES
      USE GALAHAD_LMS_MATLAB_TYPES
      USE GALAHAD_SHA_MATLAB_TYPES
      USE GALAHAD_TRU_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: TRU_matlab_control_set, TRU_matlab_control_get,                &
                TRU_matlab_inform_create, TRU_matlab_inform_get

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

      TYPE, PUBLIC :: TRU_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: TRU_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, cg_iter, f_eval, g_eval, h_eval
        mwPointer :: factorization_max, factorization_status
        mwPointer :: max_entries_factors, factorization_integer
        mwPointer :: factorization_real, factorization_average
        mwPointer :: obj, norm_g, radius
        mwPointer :: time
        TYPE ( TRU_time_pointer_type ) :: time_pointer
        TYPE ( TRS_pointer_type ) :: TRS_pointer
        TYPE ( GLTR_pointer_type ) :: GLTR_pointer
        TYPE ( PSLS_pointer_type ) :: PSLS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer_prec
        TYPE ( SHA_pointer_type ) :: SHA_pointer
      END TYPE

    CONTAINS

!-*-*-  T R U _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE TRU_matlab_control_set( ps, TRU_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to TRU

!  Arguments

!  ps - given pointer to the structure
!  TRU_control - TRU control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( TRU_control_type ) :: TRU_control

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
                                 pc, TRU_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, TRU_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, TRU_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, TRU_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, TRU_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, TRU_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, TRU_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, TRU_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, TRU_control%alive_file, len )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, TRU_control%non_monotone )
        CASE( 'model' )
          CALL MATLAB_get_value( ps, 'model',                                  &
                                 pc, TRU_control%model )
        CASE( 'norm' )
          CALL MATLAB_get_value( ps, 'norm',                                   &
                                 pc, TRU_control%norm )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, TRU_control%semi_bandwidth )
        CASE( 'lbfgs_vectors' )
          CALL MATLAB_get_value( ps, 'lbfgs_vectors',                          &
                                 pc, TRU_control%lbfgs_vectors )
        CASE( 'max_dxg' )
          CALL MATLAB_get_value( ps, 'max_dxg',                                &
                                 pc, TRU_control%max_dxg )
        CASE( 'icfs_vectors' )
          CALL MATLAB_get_value( ps, 'icfs_vectors',                           &
                                 pc, TRU_control%icfs_vectors )
        CASE( 'mi28_lsize' )
          CALL MATLAB_get_value( ps, 'mi28_lsize',                             &
                                 pc, TRU_control%mi28_lsize )
        CASE( 'mi28_rsize' )
          CALL MATLAB_get_value( ps, 'mi28_rsize',                             &
                                 pc, TRU_control%mi28_rsize )
        CASE( 'advanced_start' )
          CALL MATLAB_get_value( ps, 'advanced_start',                         &
                                 pc, TRU_control%advanced_start )
        CASE( 'stop_g_absolute' )
          CALL MATLAB_get_value( ps, 'stop_g_absolute',                        &
                                 pc, TRU_control%stop_g_absolute )
        CASE( 'stop_g_relative' )
          CALL MATLAB_get_value( ps, 'stop_g_relative',                        &
                                 pc, TRU_control%stop_g_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, TRU_control%stop_s )
        CASE( 'initial_radius' )
          CALL MATLAB_get_value( ps, 'initial_radius',                         &
                                 pc, TRU_control%initial_radius )
        CASE( 'maximum_radius' )
          CALL MATLAB_get_value( ps, 'maximum_radius',                         &
                                 pc, TRU_control%maximum_radius )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, TRU_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, TRU_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, TRU_control%eta_too_successful )
        CASE( 'radius_increase' )
          CALL MATLAB_get_value( ps, 'radius_increase',                        &
                                 pc, TRU_control%radius_increase )
        CASE( 'radius_reduce' )
          CALL MATLAB_get_value( ps, 'radius_reduce',                          &
                                 pc, TRU_control%radius_reduce )
        CASE( 'radius_reduce_max' )
          CALL MATLAB_get_value( ps, 'radius_reduce_max',                      &
                                 pc, TRU_control%radius_reduce_max )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, TRU_control%obj_unbounded )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, TRU_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, TRU_control%clock_time_limit )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, TRU_control%hessian_available )
        CASE( 'subproblem_direct' )
          CALL MATLAB_get_value( ps, 'subproblem_direct',                      &
                                 pc, TRU_control%subproblem_direct )
        CASE( 'retrospective_trust_region' )
          CALL MATLAB_get_value( ps, 'retrospective_trust_region',             &
                                 pc, TRU_control%retrospective_trust_region )
        CASE( 'renormalize_radius' )
          CALL MATLAB_get_value( ps, 'renormalize_radius',                     &
                                 pc, TRU_control%renormalize_radius )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, TRU_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, TRU_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, TRU_control%prefix, len )
        CASE( 'TRS_control' )
          pc = mxGetField( ps, 1_mwi_, 'TRS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component TRS_control must be a structure' )
          CALL TRS_matlab_control_set( pc, TRU_control%TRS_control, len )
        CASE( 'GLTR_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLTR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLTR_control must be a structure' )
          CALL GLTR_matlab_control_set( pc, TRU_control%GLTR_control, len )
        CASE( 'PSLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'PSLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component PSLS_control must be a structure' )
          CALL PSLS_matlab_control_set( pc, TRU_control%PSLS_control, len )
        CASE( 'LMS_control' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component LMS_control must be a structure' )
          CALL LMS_matlab_control_set( pc, TRU_control%LMS_control, len )
        CASE( 'LMS_control_prec' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control_prec' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt(' component LMS_control_prec must be a structure')
          CALL LMS_matlab_control_set( pc, TRU_control%LMS_control_prec, len )
        CASE( 'SHA_control' )
          pc = mxGetField( ps, 1_mwi_, 'SHA_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SHA_control must be a structure' )
          CALL SHA_matlab_control_set( pc, TRU_control%SHA_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine TRU_matlab_control_set

      END SUBROUTINE TRU_matlab_control_set

!-*-  T R U _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE TRU_matlab_control_get( struct, TRU_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to TRU

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRU_control - TRU control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRU_control_type ) :: TRU_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 46
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
         'initial_radius                 ', 'maximum_radius                 ', &
         'eta_successful                 ', 'eta_very_successful            ', &
         'eta_too_successful             ', 'radius_increase                ', &
         'radius_reduce                  ', 'radius_reduce_max              ', &
         'obj_unbounded                  ', 'cpu_time_limit                 ', &
         'clock_time_limit               ', 'hessian_available              ', &
         'subproblem_direct              ', 'retrospective_trust_region     ', &
         'renormalize_radius             ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'TRS_control                    ', 'GLTR_control                   ', &
         'PSLS_control                   ', 'LMS_control                    ', &
         'LMS_control_prec               ', 'SHA_control                    ' /)

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
                                  TRU_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  TRU_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  TRU_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  TRU_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  TRU_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  TRU_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  TRU_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  TRU_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  TRU_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  TRU_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  TRU_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'model',                            &
                                  TRU_control%model )
      CALL MATLAB_fill_component( pointer, 'norm',                             &
                                  TRU_control%norm )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  TRU_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'lbfgs_vectors',                    &
                                  TRU_control%lbfgs_vectors )
      CALL MATLAB_fill_component( pointer, 'max_dxg',                          &
                                  TRU_control%max_dxg )
      CALL MATLAB_fill_component( pointer, 'icfs_vectors',                     &
                                  TRU_control%icfs_vectors )
      CALL MATLAB_fill_component( pointer, 'mi28_lsize',                       &
                                  TRU_control%mi28_lsize )
      CALL MATLAB_fill_component( pointer, 'mi28_rsize',                       &
                                  TRU_control%mi28_rsize )
      CALL MATLAB_fill_component( pointer, 'advanced_start',                   &
                                  TRU_control%advanced_start )
      CALL MATLAB_fill_component( pointer, 'stop_g_absolute',                  &
                                  TRU_control%stop_g_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_g_relative',                  &
                                  TRU_control%stop_g_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  TRU_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'initial_radius',                   &
                                  TRU_control%initial_radius )
      CALL MATLAB_fill_component( pointer, 'maximum_radius',                   &
                                  TRU_control%maximum_radius )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  TRU_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  TRU_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  TRU_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'radius_increase',                  &
                                  TRU_control%radius_increase )
      CALL MATLAB_fill_component( pointer, 'radius_reduce',                    &
                                  TRU_control%radius_reduce )
      CALL MATLAB_fill_component( pointer, 'radius_reduce_max',                &
                                  TRU_control%radius_reduce_max )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  TRU_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  TRU_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  TRU_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  TRU_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'subproblem_direct',                &
                                  TRU_control%subproblem_direct )
      CALL MATLAB_fill_component( pointer, 'retrospective_trust_region',       &
                                  TRU_control%retrospective_trust_region )
      CALL MATLAB_fill_component( pointer, 'renormalize_radius',               &
                                  TRU_control%renormalize_radius )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  TRU_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  TRU_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  TRU_control%prefix )

!  create the components of sub-structure TRS_control

      CALL TRS_matlab_control_get( pointer, TRU_control%TRS_control,           &
                                   'TRS_control' )

!  create the components of sub-structure GLTR_control

      CALL GLTR_matlab_control_get( pointer, TRU_control%GLTR_control,         &
                                  'GLTR_control' )

!  create the components of sub-structure PSLS_control

      CALL PSLS_matlab_control_get( pointer, TRU_control%PSLS_control,         &
                                  'PSLS_control' )

!  create the components of sub-structure LMS_control

      CALL LMS_matlab_control_get( pointer, TRU_control%LMS_control,           &
                                  'LMS_control' )

!  create the components of sub-structure LMS_control_prec

      CALL LMS_matlab_control_get( pointer, TRU_control%LMS_control_prec,      &
                                  'LMS_control_prec' )

!  create the components of sub-structure SHA_control

      CALL SHA_matlab_control_get( pointer, TRU_control%SHA_control,           &
                                  'SHA_control' )

      RETURN

!  End of subroutine TRU_matlab_control_get

      END SUBROUTINE TRU_matlab_control_get

!-*-  T R U _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE TRU_matlab_inform_create( struct, TRU_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold TRU_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRU_pointer - TRU pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRU_pointer_type ) :: TRU_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 24
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'cg_iter              ', 'f_eval               ',                   &
           'g_eval               ', 'h_eval               ',                   &
           'factorization_status ', 'factorization_max    ',                   &
           'max_entries_factors  ', 'factorization_integer',                   &
           'factorization_real   ', 'factorization_average',                   &
           'obj                  ', 'norm_g               ',                   &
           'radius               ', 'time                 ',                   &
           'TRS_inform           ', 'GLTR_inform          ',                   &
           'PSLS_inform          ', 'LMS_inform           ',                   &
           'LMS_inform_prec      ', 'SHA_inform           ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_preprocess     ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, TRU_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        TRU_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'status', TRU_pointer%status )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
         'alloc_status', TRU_pointer%alloc_status )
      CALL MATLAB_create_char_component( TRU_pointer%pointer,                  &
        'bad_alloc', TRU_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'iter', TRU_pointer%iter )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'cg_iter', TRU_pointer%cg_iter )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'f_eval', TRU_pointer%f_eval )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'g_eval', TRU_pointer%g_eval )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'h_eval', TRU_pointer%h_eval )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'factorization_status', TRU_pointer%factorization_status )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'factorization_max', TRU_pointer%factorization_max )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'max_entries_factors', TRU_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'factorization_integer', TRU_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( TRU_pointer%pointer,               &
        'factorization_real', TRU_pointer%factorization_real )
      CALL MATLAB_create_real_component( TRU_pointer%pointer,                  &
        'factorization_average', TRU_pointer%factorization_average )
      CALL MATLAB_create_real_component( TRU_pointer%pointer,                  &
        'obj', TRU_pointer%obj )
      CALL MATLAB_create_real_component( TRU_pointer%pointer,                  &
        'norm_g', TRU_pointer%norm_g )
      CALL MATLAB_create_real_component( TRU_pointer%pointer,                  &
        'radius', TRU_pointer%radius )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( TRU_pointer%pointer,                    &
        'time', TRU_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'total', TRU_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'preprocess', TRU_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'analyse', TRU_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'factorize', TRU_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'solve', TRU_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'clock_total', TRU_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'clock_preprocess', TRU_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'clock_analyse', TRU_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'clock_factorize', TRU_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( TRU_pointer%time_pointer%pointer,     &
        'clock_solve', TRU_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure TRS_inform

      CALL TRS_matlab_inform_create( TRU_pointer%pointer,                      &
                                     TRU_pointer%TRS_pointer, 'TRS_inform' )

!  Define the components of sub-structure GLTR_inform

      CALL GLTR_matlab_inform_create( TRU_pointer%pointer,                     &
                                      TRU_pointer%GLTR_pointer, 'GLTR_inform' )

!  Define the components of sub-structure PSLS_inform

      CALL PSLS_matlab_inform_create( TRU_pointer%pointer,                     &
                                      TRU_pointer%PSLS_pointer, 'PSLS_inform' )

!  Define the components of sub-structure LMS_inform

      CALL LMS_matlab_inform_create( TRU_pointer%pointer,                      &
                                     TRU_pointer%LMS_pointer, 'LMS_inform' )

!  Define the components of sub-structure LMS_inform_prec

      CALL LMS_matlab_inform_create( TRU_pointer%pointer,                      &
                                     TRU_pointer%LMS_pointer_prec,             &
                                     'LMS_inform_prec')

!  Define the components of sub-structure SHA_inform

      CALL SHA_matlab_inform_create( TRU_pointer%pointer,                      &
                                     TRU_pointer%SHA_pointer, 'SHA_inform' )

      RETURN

!  End of subroutine TRU_matlab_inform_create

      END SUBROUTINE TRU_matlab_inform_create

!-*-*-  T R U _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE TRU_matlab_inform_get( TRU_inform, TRU_pointer )

!  --------------------------------------------------------------

!  Set TRU_inform values from matlab pointers

!  Arguments

!  TRU_inform - TRU inform structure
!  TRU_pointer - TRU pointer structure

!  --------------------------------------------------------------

      TYPE ( TRU_inform_type ) :: TRU_inform
      TYPE ( TRU_pointer_type ) :: TRU_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( TRU_inform%status,                              &
                               mxGetPr( TRU_pointer%status ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%alloc_status,                        &
                               mxGetPr( TRU_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( TRU_pointer%pointer,                            &
                               'bad_alloc', TRU_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( TRU_inform%iter,                                &
                               mxGetPr( TRU_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%cg_iter,                             &
                               mxGetPr( TRU_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%f_eval,                              &
                               mxGetPr( TRU_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%g_eval,                              &
                               mxGetPr( TRU_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%h_eval,                              &
                               mxGetPr( TRU_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%factorization_status,                &
                               mxGetPr( TRU_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%factorization_max,                   &
                               mxGetPr( TRU_pointer%factorization_max ) )
      CALL galmxCopyLongToPtr( TRU_inform%max_entries_factors,                 &
                               mxGetPr( TRU_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%factorization_integer,               &
                               mxGetPr( TRU_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%factorization_real,                  &
                               mxGetPr( TRU_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%factorization_average,               &
                               mxGetPr( TRU_pointer%factorization_average ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%obj,                                 &
                               mxGetPr( TRU_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%norm_g,                              &
                               mxGetPr( TRU_pointer%norm_g ) )
      CALL MATLAB_copy_to_ptr( TRU_inform%radius,                              &
                               mxGetPr( TRU_pointer%radius ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%total, wp ),              &
                           mxGetPr( TRU_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%preprocess, wp ),         &
                           mxGetPr( TRU_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%analyse, wp ),            &
                           mxGetPr( TRU_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%factorize, wp ),          &
                           mxGetPr( TRU_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%solve, wp ),              &
                           mxGetPr( TRU_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%clock_total, wp ),        &
                           mxGetPr( TRU_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%clock_preprocess, wp ),   &
                           mxGetPr( TRU_pointer%time_pointer%clock_preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%clock_analyse, wp ),      &
                           mxGetPr( TRU_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%clock_factorize, wp ),    &
                           mxGetPr( TRU_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRU_inform%time%clock_solve, wp ),        &
                           mxGetPr( TRU_pointer%time_pointer%clock_solve ) )

!  direct subproblem solver components

      CALL TRS_matlab_inform_get( TRU_inform%TRS_inform,                       &
                                  TRU_pointer%TRS_pointer )

!  iterative subproblem solver components

      CALL GLTR_matlab_inform_get( TRU_inform%GLTR_inform,                     &
                                   TRU_pointer%GLTR_pointer )

!  linear system solver components

      CALL PSLS_matlab_inform_get( TRU_inform%PSLS_inform,                     &
                                   TRU_pointer%PSLS_pointer )

!  limited memory solver components

      CALL LMS_matlab_inform_get( TRU_inform%LMS_inform,                       &
                                  TRU_pointer%LMS_pointer )

!  limited memory preconditioner components

      CALL LMS_matlab_inform_get( TRU_inform%LMS_inform_prec,                  &
                                  TRU_pointer%LMS_pointer_prec )

!  sparse Hessian approximation components

      CALL SHA_matlab_inform_get( TRU_inform%SHA_inform,                       &
                                  TRU_pointer%SHA_pointer )

      RETURN

!  End of subroutine TRU_matlab_inform_get

      END SUBROUTINE TRU_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ T R U _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_TRU_MATLAB_TYPES
