#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.2 - 06/03/2019 AT 10:40 GMT.

!-*-*-*-  G A L A H A D _ N L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.2. March 6th, 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_NLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to NLS

      USE GALAHAD_MATLAB
      USE GALAHAD_PSLS_MATLAB_TYPES
      USE GALAHAD_GLRT_MATLAB_TYPES
      USE GALAHAD_RQS_MATLAB_TYPES
      USE GALAHAD_BSC_MATLAB_TYPES
      USE GALAHAD_ROOTS_MATLAB_TYPES
      USE GALAHAD_NLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: NLS_matlab_control_set, NLS_matlab_control_get,                &
                NLS_matlab_inform_create, NLS_matlab_inform_get

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

      TYPE, PUBLIC :: NLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE NLS_time_pointer_type

      TYPE, PUBLIC :: NLS_subproblem_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, cg_iter, c_eval, j_eval, h_eval
        mwPointer :: factorization_max, factorization_status
        mwPointer :: max_entries_factors, factorization_integer
        mwPointer :: factorization_real, factorization_average
        mwPointer :: obj, norm_c, norm_g, weight
        mwPointer :: time
        TYPE ( NLS_time_pointer_type ) :: time_pointer
        TYPE ( RQS_pointer_type ) :: RQS_pointer
        TYPE ( GLRT_pointer_type ) :: GLRT_pointer
        TYPE ( PSLS_pointer_type ) :: PSLS_pointer
        TYPE ( BSC_pointer_type ) :: BSC_pointer
        TYPE ( ROOTS_pointer_type ) :: ROOTS_pointer
      END TYPE NLS_subproblem_pointer_type

      TYPE, PUBLIC, EXTENDS( NLS_subproblem_pointer_type ) :: NLS_pointer_type
        TYPE ( NLS_subproblem_pointer_type ) :: subproblem_pointer
      END TYPE NLS_pointer_type

   CONTAINS

!- N L S _ M A T L A B _ C O N T R O L _ S E T _ M A I N  S U B R O U T I N E -

      SUBROUTINE NLS_matlab_control_set_main( ps, NLS_control, len, nfields )

!  ----------------------------------------------------------------------------

!  Set matlab control arguments from values provided to NLS

!  Arguments

!  ps - given pointer to the structure
!  NLS_control - NLS control structure
!  len - length of any character component
!  nfields - only the first nfields fields in the structure will be considered

!  ----------------------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
!     TYPE ( NLS_control_type ) :: NLS_control
      TYPE ( NLS_subproblem_control_type ) :: NLS_control
      INTEGER :: nfields

!  local variables

      INTEGER :: j
      mwPointer :: pc, mxGetField
      LOGICAL :: mxIsStruct
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

!     nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, NLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, NLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, NLS_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, NLS_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, NLS_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, NLS_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, NLS_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, NLS_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, NLS_control%alive_file, len )
        CASE( 'jacobian_available' )
          CALL MATLAB_get_value( ps, 'jacobian_available',                     &
                                 pc, NLS_control%jacobian_available )
        CASE( 'hessian_available' )
          CALL MATLAB_get_value( ps, 'hessian_available',                      &
                                 pc, NLS_control%hessian_available )
        CASE( 'model' )
          CALL MATLAB_get_value( ps, 'model',                                  &
                                 pc, NLS_control%model )
        CASE( 'norm' )
          CALL MATLAB_get_value( ps, 'norm',                                   &
                                 pc, NLS_control%norm )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, NLS_control%non_monotone )
        CASE( 'weight_update_strategy' )
          CALL MATLAB_get_value( ps, 'weight_update_strategy',                 &
                                 pc, NLS_control%weight_update_strategy )
        CASE( 'stop_c_absolute' )
          CALL MATLAB_get_value( ps, 'stop_c_absolute',                        &
                                 pc, NLS_control%stop_c_absolute )
        CASE( 'stop_c_relative' )
          CALL MATLAB_get_value( ps, 'stop_c_relative',                        &
                                 pc, NLS_control%stop_c_relative )
        CASE( 'stop_g_absolute' )
          CALL MATLAB_get_value( ps, 'stop_g_absolute',                        &
                                 pc, NLS_control%stop_g_absolute )
        CASE( 'stop_g_relative' )
          CALL MATLAB_get_value( ps, 'stop_g_relative',                        &
                                 pc, NLS_control%stop_g_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, NLS_control%stop_s )
        CASE( 'power' )
          CALL MATLAB_get_value( ps, 'power',                                  &
                                 pc, NLS_control%power )
        CASE( 'initial_weight' )
          CALL MATLAB_get_value( ps, 'initial_weight',                         &
                                 pc, NLS_control%initial_weight )
        CASE( 'minimum_weight' )
          CALL MATLAB_get_value( ps, 'minimum_weight',                         &
                                 pc, NLS_control%minimum_weight )
        CASE( 'initial_inner_weight' )
          CALL MATLAB_get_value( ps, 'initial_inner_weight',                   &
                                 pc, NLS_control%initial_weight )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, NLS_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, NLS_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, NLS_control%eta_too_successful )
        CASE( 'weight_decrease_min' )
          CALL MATLAB_get_value( ps, 'weight_decrease_min',                    &
                                 pc, NLS_control%weight_decrease_min )
        CASE( 'weight_decrease' )
          CALL MATLAB_get_value( ps, 'weight_decrease',                        &
                                 pc, NLS_control%weight_decrease )
        CASE( 'weight_increase' )
          CALL MATLAB_get_value( ps, 'weight_increase',                        &
                                 pc, NLS_control%weight_increase )
        CASE( 'weight_increase_max' )
          CALL MATLAB_get_value( ps, 'weight_increase_max',                    &
                                 pc, NLS_control%weight_increase_max )
        CASE( 'reduce_gap' )
          CALL MATLAB_get_value( ps, 'reduce_gap',                             &
                                 pc, NLS_control%reduce_gap )
        CASE( 'tiny_gap' )
          CALL MATLAB_get_value( ps, 'tiny_gap',                               &
                                 pc, NLS_control%tiny_gap )
        CASE( 'large_root' )
          CALL MATLAB_get_value( ps, 'large_root',                             &
                                 pc, NLS_control%large_root )
        CASE( 'switch_to_newton' )
          CALL MATLAB_get_value( ps, 'switch_to_newton',                       &
                                 pc, NLS_control%switch_to_newton )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, NLS_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, NLS_control%clock_time_limit )
        CASE( 'subproblem_direct' )
          CALL MATLAB_get_value( ps, 'subproblem_direct',                      &
                                 pc, NLS_control%subproblem_direct )
        CASE( 'renormalize_weight' )
          CALL MATLAB_get_value( ps, 'renormalize_weight',                     &
                                 pc, NLS_control%renormalize_weight )
        CASE( 'magic_step' )
          CALL MATLAB_get_value( ps, 'magic_step',                             &
                                 pc, NLS_control%magic_step )
        CASE( 'print_obj' )
          CALL MATLAB_get_value( ps, 'print_obj',                              &
                                 pc, NLS_control%print_obj )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, NLS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, NLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, NLS_control%prefix, len )
        CASE( 'RQS_control' )
          pc = mxGetField( ps, 1_mwi_, 'RQS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component RQS_control must be a structure' )
          CALL RQS_matlab_control_set( pc, NLS_control%RQS_control, len )
        CASE( 'GLRT_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLRT_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLRT_control must be a structure' )
          CALL GLRT_matlab_control_set( pc, NLS_control%GLRT_control, len )
        CASE( 'PSLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'PSLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component PSLS_control must be a structure' )
          CALL PSLS_matlab_control_set( pc, NLS_control%PSLS_control, len )
        CASE( 'BSC_control' )
          pc = mxGetField( ps, 1_mwi_, 'BSC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component BSC_control must be a structure' )
          CALL BSC_matlab_control_set( pc, NLS_control%BSC_control, len )
        CASE( 'ROOTS_control' )
          pc = mxGetField( ps, 1_mwi_, 'ROOTS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component ROOTS_control must be a structure' )
          CALL ROOTS_matlab_control_set( pc, NLS_control%ROOTS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine NLS_matlab_control_set_main

      END SUBROUTINE NLS_matlab_control_set_main

!-*-*-  N L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE NLS_matlab_control_set( ps, NLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to NLS

!  Arguments

!  ps - given pointer to the structure
!  NLS_control - NLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( NLS_control_type ) :: NLS_control

!  local variables

      INTEGER :: nfields
      mwPointer :: pc, mxGetField
      mwSize :: mxGetNumberOfFields
      LOGICAL :: mxIsStruct
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      CALL NLS_matlab_control_set_main( ps,                                    &
             NLS_control%NLS_subproblem_control_type, len, nfields - 1 )

      name = mxGetFieldNameByNumber( ps, nfields )
      IF ( TRIM( name ) == 'subproblem_control' ) THEN
        pc = mxGetField( ps, 1_mwi_, 'subproblem_control' )
        IF ( .NOT. mxIsStruct( pc ) )                                          &
          CALL mexErrMsgTxt(                                                   &
            ' component subproblem_control must be a structure' )
        CALL NLS_matlab_control_set_main( pc,                                  &
          NLS_control%subproblem_control, len, nfields - 1 )
      END IF

      RETURN

!  End of subroutine NLS_matlab_control_set

      END SUBROUTINE NLS_matlab_control_set

!- N L S _ M A T L A B _ C O N T R O L _ G E T _ M A I N  S U B R O U T I N E -

      SUBROUTINE NLS_matlab_control_get_main( struct, NLS_control,             &
                                              ninform, finform, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to NLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NLS_control - NLS control structure
!  name - name of component of the structure
!  ninform - number of components
!  finform - character array of names of components

!  --------------------------------------------------------------

      mwPointer :: struct
!     TYPE ( NLS_control_type ) :: NLS_control
      TYPE ( NLS_subproblem_control_type ) :: NLS_control
      INTEGER * 4 :: ninform
      CHARACTER ( LEN = 31 ) :: finform( ninform )
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

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
                                  NLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  NLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  NLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  NLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  NLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  NLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  NLS_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  NLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  NLS_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  NLS_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'jacobian_available',               &
                                  NLS_control%jacobian_available )
      CALL MATLAB_fill_component( pointer, 'hessian_available',                &
                                  NLS_control%hessian_available )
      CALL MATLAB_fill_component( pointer, 'model',                            &
                                  NLS_control%model )
      CALL MATLAB_fill_component( pointer, 'norm',                             &
                                  NLS_control%norm )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  NLS_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'weight_update_strategy',           &
                                  NLS_control%weight_update_strategy )
      CALL MATLAB_fill_component( pointer, 'stop_c_absolute',                  &
                                  NLS_control%stop_c_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_c_relative',                  &
                                  NLS_control%stop_c_relative )
      CALL MATLAB_fill_component( pointer, 'stop_g_absolute',                  &
                                  NLS_control%stop_g_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_g_relative',                  &
                                  NLS_control%stop_g_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  NLS_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'power',                            &
                                  NLS_control%power )
      CALL MATLAB_fill_component( pointer, 'initial_weight',                   &
                                  NLS_control%initial_weight )
      CALL MATLAB_fill_component( pointer, 'minimum_weight',                   &
                                  NLS_control%minimum_weight )
      CALL MATLAB_fill_component( pointer, 'initial_inner_weight',             &
                                  NLS_control%initial_inner_weight )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  NLS_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  NLS_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  NLS_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'weight_decrease_min',              &
                                  NLS_control%weight_decrease_min )
      CALL MATLAB_fill_component( pointer, 'weight_decrease',                  &
                                  NLS_control%weight_decrease )
      CALL MATLAB_fill_component( pointer, 'weight_increase',                  &
                                  NLS_control%weight_increase )
      CALL MATLAB_fill_component( pointer, 'weight_increase_max',              &
                                  NLS_control%weight_increase_max )
      CALL MATLAB_fill_component( pointer, 'reduce_gap',                       &
                                  NLS_control%reduce_gap )
      CALL MATLAB_fill_component( pointer, 'tiny_gap',                         &
                                  NLS_control%tiny_gap )
      CALL MATLAB_fill_component( pointer, 'large_root',                       &
                                  NLS_control%large_root )
      CALL MATLAB_fill_component( pointer, 'switch_to_newton',                 &
                                  NLS_control%switch_to_newton )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  NLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  NLS_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'subproblem_direct',                &
                                  NLS_control%subproblem_direct )
      CALL MATLAB_fill_component( pointer, 'renormalize_weight',               &
                                  NLS_control%renormalize_weight )
      CALL MATLAB_fill_component( pointer, 'magic_step',                       &
                                  NLS_control%magic_step )
      CALL MATLAB_fill_component( pointer, 'print_obj',                        &
                                  NLS_control%print_obj )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  NLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  NLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  NLS_control%prefix )

!  create the components of sub-structure RQS_control

      CALL RQS_matlab_control_get( pointer, NLS_control%RQS_control,           &
                                   'RQS_control' )

!  create the components of sub-structure GLRT_control

      CALL GLRT_matlab_control_get( pointer, NLS_control%GLRT_control,         &
                                  'GLRT_control' )

!  create the components of sub-structure PSLS_control

      CALL PSLS_matlab_control_get( pointer, NLS_control%PSLS_control,         &
                                  'PSLS_control' )

!  create the components of sub-structure BSC_control

      CALL BSC_matlab_control_get( pointer, NLS_control%BSC_control,           &
                                  'BSC_control' )

!  create the components of sub-structure ROOTS_control

      CALL ROOTS_matlab_control_get( pointer, NLS_control%ROOTS_control,       &
                                  'ROOTS_control' )

      RETURN

!  End of subroutine NLS_matlab_control_get_main

      END SUBROUTINE NLS_matlab_control_get_main

!-*-  N L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE NLS_matlab_control_get( struct, NLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to NLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NLS_control - NLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NLS_control_type ) :: NLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 50
      INTEGER * 4, PARAMETER :: ninformm1 = ninform - 1
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'alive_unit                     ', &
         'alive_file                     ', 'jacobian_available             ', &
         'hessian_available              ', 'model                          ', &
         'norm                           ', 'non_monotone                   ', &
         'weight_update_strategy         ', 'stop_c_absolute                ', &
         'stop_c_relative                ', 'stop_g_absolute                ', &
         'stop_g_relative                ', 'stop_s                         ', &
         'power                          ', 'initial_weight                 ', &
         'minimum_weight                 ', 'initial_inner_weight           ', &
         'eta_successful                 ', 'eta_very_successful            ', &
         'eta_too_successful             ', 'weight_decrease_min            ', &
         'weight_decrease                ', 'weight_increase                ', &
         'weight_increase_max            ', 'reduce_gap                     ', &
         'tiny_gap                       ', 'large_root                     ', &
         'switch_to_newton               ', 'cpu_time_limit                 ', &
         'clock_time_limit               ', 'subproblem_direct              ', &
         'renormalize_weight             ', 'magic_step                     ', &
         'print_obj                      ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'RQS_control                    ', 'GLRT_control                   ', &
         'PSLS_control                   ', 'BSC_control                    ', &
         'ROOTS_control                  ', 'subproblem_control             ' /)

!  create and get the components

      CALL NLS_matlab_control_get_main( struct,                                &
             NLS_control%NLS_subproblem_control_type, ninform, finform, name )

!  create and get the components of the sub-structure subproblem_control

      pointer = struct
      CALL NLS_matlab_control_get_main( pointer,                               &
             NLS_control%subproblem_control, ninformm1, finform,               &
             'subproblem_control' )

      RETURN

!  End of subroutine NLS_matlab_control_get

      END SUBROUTINE NLS_matlab_control_get

! N L S _ M A T L A B _ I N F O R M _ C R E A T E _ M A I N  S U B R O U T I N E

      SUBROUTINE NLS_matlab_inform_create_main( struct, NLS_pointer,           &
                                ninform, finform, t_ninform, t_finform, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold NLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NLS_pointer - NLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)
!  ninform - number of components
!  finform - character array of names of components
!  ninform - number of timing components
!  finform - character array of names of timing components

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NLS_subproblem_pointer_type ) :: NLS_pointer
      INTEGER * 4 :: ninform, t_ninform
      CHARACTER ( LEN = 21 ) :: finform( ninform ), t_finform( t_ninform )
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, NLS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        NLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'status', NLS_pointer%status )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
         'alloc_status', NLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( NLS_pointer%pointer,                  &
        'bad_alloc', NLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'iter', NLS_pointer%iter )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'cg_iter', NLS_pointer%cg_iter )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'c_eval', NLS_pointer%c_eval )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'j_eval', NLS_pointer%j_eval )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'h_eval', NLS_pointer%h_eval )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'factorization_status', NLS_pointer%factorization_status )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'factorization_max', NLS_pointer%factorization_max )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'max_entries_factors', NLS_pointer%max_entries_factors )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'factorization_integer', NLS_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( NLS_pointer%pointer,               &
        'factorization_real', NLS_pointer%factorization_real )
      CALL MATLAB_create_real_component( NLS_pointer%pointer,                  &
        'factorization_average', NLS_pointer%factorization_average )
      CALL MATLAB_create_real_component( NLS_pointer%pointer,                  &
        'obj', NLS_pointer%obj )
      CALL MATLAB_create_real_component( NLS_pointer%pointer,                  &
        'norm_c', NLS_pointer%norm_c )
      CALL MATLAB_create_real_component( NLS_pointer%pointer,                  &
        'norm_g', NLS_pointer%norm_g )
      CALL MATLAB_create_real_component( NLS_pointer%pointer,                  &
        'weight', NLS_pointer%weight )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( NLS_pointer%pointer,                    &
        'time', NLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'total', NLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'preprocess', NLS_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'analyse', NLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'factorize', NLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'solve', NLS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'clock_total', NLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'clock_preprocess', NLS_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'clock_analyse', NLS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'clock_factorize', NLS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( NLS_pointer%time_pointer%pointer,     &
        'clock_solve', NLS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure RQS_inform

      CALL RQS_matlab_inform_create( NLS_pointer%pointer,                      &
                                     NLS_pointer%RQS_pointer, 'RQS_inform' )

!  Define the components of sub-structure GLRT_inform

      CALL GLRT_matlab_inform_create( NLS_pointer%pointer,                     &
                                      NLS_pointer%GLRT_pointer, 'GLRT_inform' )

!  Define the components of sub-structure PSLS_inform

      CALL PSLS_matlab_inform_create( NLS_pointer%pointer,                     &
                                      NLS_pointer%PSLS_pointer, 'PSLS_inform' )

!  Define the components of sub-structure BSC_inform

      CALL BSC_matlab_inform_create( NLS_pointer%pointer,                      &
                                     NLS_pointer%BSC_pointer, 'BSC_inform' )

!  Define the components of sub-structure ROOTS_inform

      CALL ROOTS_matlab_inform_create( NLS_pointer%pointer,                    &
                                       NLS_pointer%ROOTS_pointer,              &
                                       'ROOTS_inform' )

      RETURN

!  End of subroutine NLS_matlab_inform_create_main

      END SUBROUTINE NLS_matlab_inform_create_main

!-*-  N L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE NLS_matlab_inform_create( struct, NLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold NLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NLS_pointer - NLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NLS_pointer_type ) :: NLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      INTEGER * 4, PARAMETER :: ninform = 25
      INTEGER * 4, PARAMETER :: ninformm1 = ninform - 1
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'cg_iter              ', 'c_eval               ',                   &
           'j_eval               ', 'h_eval               ',                   &
           'factorization_status ', 'factorization_max    ',                   &
           'max_entries_factors  ', 'factorization_integer',                   &
           'factorization_real   ', 'factorization_average',                   &
           'obj                  ', 'norm_c               ',                   &
           'norm_g               ', 'weight               ',                   &
           'time                 ', 'RQS_inform           ',                   &
           'GLRT_inform          ', 'PSLS_inform          ',                   &
           'BSC_inform           ', 'ROOTS_inform         ',                   &
           'subproblem_inform    '                          /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_preprocess     ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  local variables

      mwPointer :: pointer

!  create and get the components

      CALL NLS_matlab_inform_create_main( struct,                              &
             NLS_pointer%NLS_subproblem_pointer_type, ninform, finform,        &
             t_ninform, t_finform, name )

!  create and get the components of the sub-structure subproblem_control

      pointer = struct
      CALL NLS_matlab_inform_create_main( pointer,                             &
             NLS_pointer%subproblem_pointer, ninformm1, finform,               &
             t_ninform, t_finform, 'subproblem_inform' )

      RETURN

!  End of subroutine NLS_matlab_inform_create

      END SUBROUTINE NLS_matlab_inform_create

!-  N L S _ M A T L A B _ I N F O R M _ G E T _ M A I N  S U B R O U T I N E   -

      SUBROUTINE NLS_matlab_inform_get_main( NLS_inform, NLS_pointer )

!  --------------------------------------------------------------

!  Set NLS_inform values from matlab pointers

!  Arguments

!  NLS_inform - NLS inform structure
!  NLS_pointer - NLS pointer structure

!  --------------------------------------------------------------

!     TYPE ( NLS_inform_type ) :: NLS_inform
      TYPE ( NLS_subproblem_inform_type ) :: NLS_inform
!     TYPE ( NLS_pointer_type ) :: NLS_pointer
      TYPE ( NLS_subproblem_pointer_type ) :: NLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( NLS_inform%status,                              &
                               mxGetPr( NLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%alloc_status,                        &
                               mxGetPr( NLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( NLS_pointer%pointer,                            &
                               'bad_alloc', NLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( NLS_inform%iter,                                &
                               mxGetPr( NLS_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%cg_iter,                             &
                               mxGetPr( NLS_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%c_eval,                              &
                               mxGetPr( NLS_pointer%c_eval ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%j_eval,                              &
                               mxGetPr( NLS_pointer%j_eval ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%h_eval,                              &
                               mxGetPr( NLS_pointer%h_eval ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%factorization_status,                &
                               mxGetPr( NLS_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%factorization_max,                   &
                               mxGetPr( NLS_pointer%factorization_max ) )
      CALL galmxCopyLongToPtr( NLS_inform%max_entries_factors,                 &
                               mxGetPr( NLS_pointer%max_entries_factors ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%factorization_integer,               &
                               mxGetPr( NLS_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%factorization_real,                  &
                               mxGetPr( NLS_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%factorization_average,               &
                               mxGetPr( NLS_pointer%factorization_average ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%obj,                                 &
                               mxGetPr( NLS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%norm_c,                              &
                               mxGetPr( NLS_pointer%norm_c ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%norm_g,                              &
                               mxGetPr( NLS_pointer%norm_g ) )
      CALL MATLAB_copy_to_ptr( NLS_inform%weight,                              &
                               mxGetPr( NLS_pointer%weight ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%total, wp ),              &
                           mxGetPr( NLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%preprocess, wp ),         &
                           mxGetPr( NLS_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%analyse, wp ),            &
                           mxGetPr( NLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%factorize, wp ),          &
                           mxGetPr( NLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%solve, wp ),              &
                           mxGetPr( NLS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%clock_total, wp ),        &
                           mxGetPr( NLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%clock_preprocess, wp ),   &
                           mxGetPr( NLS_pointer%time_pointer%clock_preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%clock_analyse, wp ),      &
                           mxGetPr( NLS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%clock_factorize, wp ),    &
                           mxGetPr( NLS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( NLS_inform%time%clock_solve, wp ),        &
                           mxGetPr( NLS_pointer%time_pointer%clock_solve ) )

!  direct subproblem solver components

      CALL RQS_matlab_inform_get( NLS_inform%RQS_inform,                       &
                                  NLS_pointer%RQS_pointer )

!  iterative subproblem solver components

      CALL GLRT_matlab_inform_get( NLS_inform%GLRT_inform,                     &
                                   NLS_pointer%GLRT_pointer )

!  linear system solver components

      CALL PSLS_matlab_inform_get( NLS_inform%PSLS_inform,                     &
                                   NLS_pointer%PSLS_pointer )

!  Schur complement constructor components

      CALL BSC_matlab_inform_get( NLS_inform%BSC_inform,                       &
                                  NLS_pointer%BSC_pointer )

!  polynomial equation solver components

      CALL ROOTS_matlab_inform_get( NLS_inform%ROOTS_inform,                   &
                                    NLS_pointer%ROOTS_pointer )

      RETURN

!  End of subroutine NLS_matlab_inform_get_main

      END SUBROUTINE NLS_matlab_inform_get_main

!-*-*-  N L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE NLS_matlab_inform_get( NLS_inform, NLS_pointer )

!  --------------------------------------------------------------

!  Set NLS_inform values from matlab pointers

!  Arguments

!  NLS_inform - NLS inform structure
!  NLS_pointer - NLS pointer structure

!  --------------------------------------------------------------

      TYPE ( NLS_inform_type ) :: NLS_inform
      TYPE ( NLS_pointer_type ) :: NLS_pointer

!  set the components

      CALL NLS_matlab_inform_get_main( NLS_inform%NLS_subproblem_inform_type,  &
                                       NLS_pointer%NLS_subproblem_pointer_type )

!  set the components of the sub-structure subproblem_inform

      CALL NLS_matlab_inform_get_main( NLS_inform%subproblem_inform,           &
                                       NLS_pointer%subproblem_pointer )

      RETURN

!  End of subroutine NLS_matlab_inform_get

      END SUBROUTINE NLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ N L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_NLS_MATLAB_TYPES
