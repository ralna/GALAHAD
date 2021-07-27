#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.3 - 22/07/2021 AT 08:40 GMT.

!-*-*-*-  G A L A H A D _ T R B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. July 21st, 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_TRB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to TRB

      USE GALAHAD_MATLAB
      USE GALAHAD_PSLS_MATLAB_TYPES
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_TRS_MATLAB_TYPES
      USE GALAHAD_LMS_MATLAB_TYPES
      USE GALAHAD_SHA_MATLAB_TYPES
      USE GALAHAD_TRB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: TRB_matlab_control_set, TRB_matlab_control_get,                &
                TRB_matlab_inform_create, TRB_matlab_inform_get

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

      TYPE, PUBLIC :: TRB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: TRB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: n_free, iter, cg_iter, cg_maxit, f_eval, g_eval, h_eval
        mwPointer :: factorization_max, factorization_status
        mwPointer :: max_entries_factors, factorization_integer
        mwPointer :: factorization_real
        mwPointer :: obj, norm_pg, radius
        mwPointer :: time
        TYPE ( TRB_time_pointer_type ) :: time_pointer
        TYPE ( TRS_pointer_type ) :: TRS_pointer
        TYPE ( GLTR_pointer_type ) :: GLTR_pointer
        TYPE ( PSLS_pointer_type ) :: PSLS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer
        TYPE ( LMS_pointer_type ) :: LMS_pointer_prec
        TYPE ( SHA_pointer_type ) :: SHA_pointer
      END TYPE

    CONTAINS

!-*-*-  T R B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE TRB_matlab_control_set( ps, TRB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to TRB

!  Arguments

!  ps - given pointer to the structure
!  TRB_control - TRB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( TRB_control_type ) :: TRB_control

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
                                 pc, TRB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, TRB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, TRB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, TRB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, TRB_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, TRB_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, TRB_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, TRB_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, TRB_control%alive_file, len )
        CASE( 'more_toraldo' )
          CALL MATLAB_get_value( ps, 'more_toraldo',                           &
                                 pc, TRB_control%more_toraldo )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, TRB_control%non_monotone )
        CASE( 'model' )
          CALL MATLAB_get_value( ps, 'model',                                  &
                                 pc, TRB_control%model )
        CASE( 'norm' )
          CALL MATLAB_get_value( ps, 'norm',                                   &
                                 pc, TRB_control%norm )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, TRB_control%semi_bandwidth )
        CASE( 'lbfgs_vectors' )
          CALL MATLAB_get_value( ps, 'lbfgs_vectors',                          &
                                 pc, TRB_control%lbfgs_vectors )
        CASE( 'max_dxg' )
          CALL MATLAB_get_value( ps, 'max_dxg',                                &
                                 pc, TRB_control%max_dxg )
        CASE( 'icfs_vectors' )
          CALL MATLAB_get_value( ps, 'icfs_vectors',                           &
                                 pc, TRB_control%icfs_vectors )
        CASE( 'mi28_lsize' )
          CALL MATLAB_get_value( ps, 'mi28_lsize',                             &
                                 pc, TRB_control%mi28_lsize )
        CASE( 'mi28_rsize' )
          CALL MATLAB_get_value( ps, 'mi28_rsize',                             &
                                 pc, TRB_control%mi28_rsize )
        CASE( 'advanced_start' )
          CALL MATLAB_get_value( ps, 'advanced_start',                         &
                                 pc, TRB_control%advanced_start )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, TRB_control%infinity )
        CASE( 'stop_pg_absolute' )
          CALL MATLAB_get_value( ps, 'stop_pg_absolute',                       &
                                 pc, TRB_control%stop_pg_absolute )
        CASE( 'stop_pg_relative' )
          CALL MATLAB_get_value( ps, 'stop_pg_relative',                       &
                                 pc, TRB_control%stop_pg_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, TRB_control%stop_s )
        CASE( 'initial_radius' )
          CALL MATLAB_get_value( ps, 'initial_radius',                         &
                                 pc, TRB_control%initial_radius )
        CASE( 'maximum_radius' )
          CALL MATLAB_get_value( ps, 'maximum_radius',                         &
                                 pc, TRB_control%maximum_radius )
        CASE( 'stop_rel_cg' )
          CALL MATLAB_get_value( ps, 'stop_rel_cg',                            &
                                 pc, TRB_control%stop_rel_cg )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, TRB_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, TRB_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, TRB_control%eta_too_successful )
        CASE( 'radius_increase' )
          CALL MATLAB_get_value( ps, 'radius_increase',                        &
                                 pc, TRB_control%radius_increase )
        CASE( 'radius_reduce' )
          CALL MATLAB_get_value( ps, 'radius_reduce',                          &
                                 pc, TRB_control%radius_reduce )
        CASE( 'radius_reduce_max' )
          CALL MATLAB_get_value( ps, 'radius_reduce_max',                      &
                                 pc, TRB_control%radius_reduce_max )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, TRB_control%obj_unbounded )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, TRB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, TRB_control%clock_time_limit )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, TRB_control%hessian_available )
        CASE( 'subproblem_direct' )
          CALL MATLAB_get_value( ps, 'subproblem_direct',                      &
                                 pc, TRB_control%subproblem_direct )
        CASE( 'retrospective_trust_region' )
          CALL MATLAB_get_value( ps, 'retrospective_trust_region',             &
                                 pc, TRB_control%retrospective_trust_region )
        CASE( 'renormalize_radius' )
          CALL MATLAB_get_value( ps, 'renormalize_radius',                     &
                                 pc, TRB_control%renormalize_radius )
        CASE( 'two_norm_tr' )
          CALL MATLAB_get_value( ps, 'two_norm_tr',                            &
                                 pc, TRB_control%two_norm_tr )
        CASE( 'exact_gcp' )
          CALL MATLAB_get_value( ps, 'exact_gcp',                              &
                                 pc, TRB_control%exact_gcp )
        CASE( 'accurate_bqp' )
          CALL MATLAB_get_value( ps, 'accurate_bqp',                           &
                                 pc, TRB_control%accurate_bqp )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, TRB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, TRB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, TRB_control%prefix, len )
        CASE( 'TRS_control' )
          pc = mxGetField( ps, 1_mwi_, 'TRS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component TRS_control must be a structure' )
          CALL TRS_matlab_control_set( pc, TRB_control%TRS_control, len )
        CASE( 'GLTR_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLTR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLTR_control must be a structure' )
          CALL GLTR_matlab_control_set( pc, TRB_control%GLTR_control, len )
        CASE( 'PSLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'PSLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component PSLS_control must be a structure' )
          CALL PSLS_matlab_control_set( pc, TRB_control%PSLS_control, len )
        CASE( 'LMS_control' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component LMS_control must be a structure' )
          CALL LMS_matlab_control_set( pc, TRB_control%LMS_control, len )
        CASE( 'LMS_control_prec' )
          pc = mxGetField( ps, 1_mwi_, 'LMS_control_prec' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt(' component LMS_control_prec must be a structure')
          CALL LMS_matlab_control_set( pc, TRB_control%LMS_control_prec, len )
        CASE( 'SHA_control' )
          pc = mxGetField( ps, 1_mwi_, 'SHA_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SHA_control must be a structure' )
          CALL SHA_matlab_control_set( pc, TRB_control%SHA_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine TRB_matlab_control_set

      END SUBROUTINE TRB_matlab_control_set

!-*-  T R B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE TRB_matlab_control_get( struct, TRB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to TRB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRB_control - TRB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRB_control_type ) :: TRB_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 52
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'alive_unit                     ', &
         'alive_file                     ', 'more_toraldo                   ', &
         'non_monotone                   ', 'model                          ', &
         'norm                           ', 'semi_bandwidth                 ', &
         'lbfgs_vectors                  ', 'max_dxg                        ', &
         'icfs_vectors                   ', 'mi28_lsize                     ', &
         'mi28_rsize                     ', 'advanced_start                 ', &
         'infinity                       ', 'stop_pg_absolute               ', &
         'stop_pg_relative               ', 'stop_s                         ', &
         'initial_radius                 ', 'maximum_radius                 ', &
         'stop_rel_cg                    ', 'eta_successful                 ', &
         'eta_very_successful            ', 'eta_too_successful             ', &
         'radius_increase                ', 'radius_reduce                  ', &
         'radius_reduce_max              ', 'obj_unbounded                  ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'hessian_available              ', 'subproblem_direct              ', &
         'retrospective_trust_region     ', 'renormalize_radius             ', &
         'two_norm_tr                    ', 'exact_gcp                      ', &
         'accurate_bqp                   ', 'space_critical                 ', &
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
                                  TRB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  TRB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  TRB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  TRB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  TRB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  TRB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  TRB_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  TRB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  TRB_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  TRB_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'more_toraldo',                     &
                                  TRB_control%more_toraldo )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  TRB_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'model',                            &
                                  TRB_control%model )
      CALL MATLAB_fill_component( pointer, 'norm',                             &
                                  TRB_control%norm )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  TRB_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'lbfgs_vectors',                    &
                                  TRB_control%lbfgs_vectors )
      CALL MATLAB_fill_component( pointer, 'max_dxg',                          &
                                  TRB_control%max_dxg )
      CALL MATLAB_fill_component( pointer, 'icfs_vectors',                     &
                                  TRB_control%icfs_vectors )
      CALL MATLAB_fill_component( pointer, 'mi28_lsize',                       &
                                  TRB_control%mi28_lsize )
      CALL MATLAB_fill_component( pointer, 'mi28_rsize',                       &
                                  TRB_control%mi28_rsize )
      CALL MATLAB_fill_component( pointer, 'advanced_start',                   &
                                  TRB_control%advanced_start )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  TRB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_pg_absolute',                 &
                                  TRB_control%stop_pg_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_pg_relative',                 &
                                  TRB_control%stop_pg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  TRB_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'initial_radius',                   &
                                  TRB_control%initial_radius )
      CALL MATLAB_fill_component( pointer, 'maximum_radius',                   &
                                  TRB_control%maximum_radius )
      CALL MATLAB_fill_component( pointer, 'stop_rel_cg',                      &
                                  TRB_control%stop_rel_cg )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  TRB_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  TRB_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  TRB_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'radius_increase',                  &
                                  TRB_control%radius_increase )
      CALL MATLAB_fill_component( pointer, 'radius_reduce',                    &
                                  TRB_control%radius_reduce )
      CALL MATLAB_fill_component( pointer, 'radius_reduce_max',                &
                                  TRB_control%radius_reduce_max )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  TRB_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  TRB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  TRB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  TRB_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'subproblem_direct',                &
                                  TRB_control%subproblem_direct )
      CALL MATLAB_fill_component( pointer, 'retrospective_trust_region',       &
                                  TRB_control%retrospective_trust_region )
      CALL MATLAB_fill_component( pointer, 'renormalize_radius',               &
                                  TRB_control%renormalize_radius )
      CALL MATLAB_fill_component( pointer, 'two_norm_tr',                      &
                                  TRB_control%two_norm_tr )
      CALL MATLAB_fill_component( pointer, 'exact_gcp',                        &
                                  TRB_control%exact_gcp )
      CALL MATLAB_fill_component( pointer, 'accurate_bqp',                     &
                                  TRB_control%accurate_bqp )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  TRB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  TRB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  TRB_control%prefix )

!  create the components of sub-structure TRS_control

      CALL TRS_matlab_control_get( pointer, TRB_control%TRS_control,           &
                                   'TRS_control' )

!  create the components of sub-structure GLTR_control

      CALL GLTR_matlab_control_get( pointer, TRB_control%GLTR_control,         &
                                  'GLTR_control' )

!  create the components of sub-structure PSLS_control

      CALL PSLS_matlab_control_get( pointer, TRB_control%PSLS_control,         &
                                  'PSLS_control' )

!  create the components of sub-structure LMS_control

      CALL LMS_matlab_control_get( pointer, TRB_control%LMS_control,           &
                                  'LMS_control' )

!  create the components of sub-structure LMS_control_prec

      CALL LMS_matlab_control_get( pointer, TRB_control%LMS_control_prec,      &
                                  'LMS_control_prec' )

!  create the components of sub-structure SHA_control

      CALL SHA_matlab_control_get( pointer, TRB_control%SHA_control,           &
                                  'SHA_control' )

      RETURN

!  End of subroutine TRB_matlab_control_get

      END SUBROUTINE TRB_matlab_control_get

!-*-  T R B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE TRB_matlab_inform_create( struct, TRB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold TRB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  TRB_pointer - TRB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( TRB_pointer_type ) :: TRB_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 25
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'n_free               ',                   &
           'iter                 ', 'cg_iter              ',                   &
           'cg_maxit             ', 'f_eval               ',                   &
           'g_eval               ', 'h_eval               ',                   &
           'factorization_status ', 'factorization_max    ',                   &
           'max_entries_factors  ', 'factorization_integer',                   &
           'factorization_real   ', 'obj                  ',                   &
           'norm_pg              ', 'radius               ',                   &
           'time                 ', 'TRS_inform           ',                   &
           'GLTR_inform          ', 'PSLS_inform          ',                   &
           'LMS_inform           ', 'LMS_inform_prec      ',                   &
           'SHA_inform           '  /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_preprocess     ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, TRB_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        TRB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'status', TRB_pointer%status )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
         'alloc_status', TRB_pointer%alloc_status )
      CALL MATLAB_create_char_component( TRB_pointer%pointer,                  &
        'bad_alloc', TRB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'n_free', TRB_pointer%n_free )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'iter', TRB_pointer%iter )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'cg_iter', TRB_pointer%cg_iter )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'cg_maxit', TRB_pointer%cg_maxit )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'f_eval', TRB_pointer%f_eval )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'g_eval', TRB_pointer%g_eval )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'h_eval', TRB_pointer%h_eval )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'factorization_status', TRB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'factorization_max', TRB_pointer%factorization_max )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'max_entries_factors', TRB_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'factorization_integer', TRB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( TRB_pointer%pointer,               &
        'factorization_real', TRB_pointer%factorization_real )
      CALL MATLAB_create_real_component( TRB_pointer%pointer,                  &
        'obj', TRB_pointer%obj )
      CALL MATLAB_create_real_component( TRB_pointer%pointer,                  &
        'norm_pg', TRB_pointer%norm_pg )
      CALL MATLAB_create_real_component( TRB_pointer%pointer,                  &
        'radius', TRB_pointer%radius )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( TRB_pointer%pointer,                    &
        'time', TRB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'total', TRB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'preprocess', TRB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'analyse', TRB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'factorize', TRB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'solve', TRB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'clock_total', TRB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'clock_preprocess', TRB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'clock_analyse', TRB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'clock_factorize', TRB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( TRB_pointer%time_pointer%pointer,     &
        'clock_solve', TRB_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure TRS_inform

      CALL TRS_matlab_inform_create( TRB_pointer%pointer,                      &
                                     TRB_pointer%TRS_pointer, 'TRS_inform' )

!  Define the components of sub-structure GLTR_inform

      CALL GLTR_matlab_inform_create( TRB_pointer%pointer,                     &
                                      TRB_pointer%GLTR_pointer, 'GLTR_inform' )

!  Define the components of sub-structure PSLS_inform

      CALL PSLS_matlab_inform_create( TRB_pointer%pointer,                     &
                                      TRB_pointer%PSLS_pointer, 'PSLS_inform' )

!  Define the components of sub-structure LMS_inform

      CALL LMS_matlab_inform_create( TRB_pointer%pointer,                      &
                                     TRB_pointer%LMS_pointer, 'LMS_inform' )

!  Define the components of sub-structure LMS_inform_prec

      CALL LMS_matlab_inform_create( TRB_pointer%pointer,                      &
                                     TRB_pointer%LMS_pointer_prec,             &
                                     'LMS_inform_prec')

!  Define the components of sub-structure SHA_inform

      CALL SHA_matlab_inform_create( TRB_pointer%pointer,                      &
                                     TRB_pointer%SHA_pointer, 'SHA_inform' )

      RETURN

!  End of subroutine TRB_matlab_inform_create

      END SUBROUTINE TRB_matlab_inform_create

!-*-*-  T R B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE TRB_matlab_inform_get( TRB_inform, TRB_pointer )

!  --------------------------------------------------------------

!  Set TRB_inform values from matlab pointers

!  Arguments

!  TRB_inform - TRB inform structure
!  TRB_pointer - TRB pointer structure

!  --------------------------------------------------------------

      TYPE ( TRB_inform_type ) :: TRB_inform
      TYPE ( TRB_pointer_type ) :: TRB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( TRB_inform%status,                              &
                               mxGetPr( TRB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%alloc_status,                        &
                               mxGetPr( TRB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( TRB_pointer%pointer,                            &
                               'bad_alloc', TRB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( TRB_inform%n_free,                              &
                               mxGetPr( TRB_pointer%n_free ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%iter,                                &
                               mxGetPr( TRB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%cg_iter,                             &
                               mxGetPr( TRB_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%cg_maxit,                            &
                               mxGetPr( TRB_pointer%cg_maxit ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%f_eval,                              &
                               mxGetPr( TRB_pointer%f_eval ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%g_eval,                              &
                               mxGetPr( TRB_pointer%g_eval ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%h_eval,                              &
                               mxGetPr( TRB_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%factorization_status,                &
                               mxGetPr( TRB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%factorization_max,                   &
                               mxGetPr( TRB_pointer%factorization_max ) )
      CALL galmxCopyLongToPtr( TRB_inform%max_entries_factors,                 &
                               mxGetPr( TRB_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%factorization_integer,               &
                               mxGetPr( TRB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%factorization_real,                  &
                               mxGetPr( TRB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%obj,                                 &
                               mxGetPr( TRB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%norm_pg,                             &
                               mxGetPr( TRB_pointer%norm_pg ) )
      CALL MATLAB_copy_to_ptr( TRB_inform%radius,                              &
                               mxGetPr( TRB_pointer%radius ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%total, wp ),              &
                           mxGetPr( TRB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%preprocess, wp ),         &
                           mxGetPr( TRB_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%analyse, wp ),            &
                           mxGetPr( TRB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%factorize, wp ),          &
                           mxGetPr( TRB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%solve, wp ),              &
                           mxGetPr( TRB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%clock_total, wp ),        &
                           mxGetPr( TRB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%clock_preprocess, wp ),   &
                           mxGetPr( TRB_pointer%time_pointer%clock_preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%clock_analyse, wp ),      &
                           mxGetPr( TRB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%clock_factorize, wp ),    &
                           mxGetPr( TRB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( TRB_inform%time%clock_solve, wp ),        &
                           mxGetPr( TRB_pointer%time_pointer%clock_solve ) )

!  direct subproblem solver components

      CALL TRS_matlab_inform_get( TRB_inform%TRS_inform,                       &
                                  TRB_pointer%TRS_pointer )

!  iterative subproblem solver components

      CALL GLTR_matlab_inform_get( TRB_inform%GLTR_inform,                     &
                                   TRB_pointer%GLTR_pointer )

!  linear system solver components

      CALL PSLS_matlab_inform_get( TRB_inform%PSLS_inform,                     &
                                   TRB_pointer%PSLS_pointer )

!  limited memory solver components

      CALL LMS_matlab_inform_get( TRB_inform%LMS_inform,                       &
                                  TRB_pointer%LMS_pointer )

!  limited memory preconditioner components

      CALL LMS_matlab_inform_get( TRB_inform%LMS_inform_prec,                  &
                                  TRB_pointer%LMS_pointer_prec )

!  sparse Hessian approximation components

      CALL SHA_matlab_inform_get( TRB_inform%SHA_inform,                       &
                                  TRB_pointer%SHA_pointer )

      RETURN

!  End of subroutine TRB_matlab_inform_get

      END SUBROUTINE TRB_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ T R B _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_TRB_MATLAB_TYPES
