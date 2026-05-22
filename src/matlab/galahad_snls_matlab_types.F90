#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-05-21 AT 11:00 GMT.

!-*-*-*-  G A L A H A D _ S N L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.5. March 6th, 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SNLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SNLS

      USE GALAHAD_MATLAB
      USE GALAHAD_SLLS_MATLAB_TYPES
      USE GALAHAD_SLLSB_MATLAB_TYPES
      USE GALAHAD_SNLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SNLS_matlab_control_set, SNLS_matlab_control_get,              &
                SNLS_matlab_inform_create, SNLS_matlab_inform_get

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

      TYPE, PUBLIC :: SNLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, slls, sllsb
        mwPointer :: clock_total, clock_slls, clock_sllsb
      END TYPE SNLS_time_pointer_type

      TYPE, PUBLIC :: SNLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, inner_iter, r_eval, jr_eval
        mwPointer :: obj, norm_r, norm_g, norm_pg, weight
        mwPointer :: time
        TYPE ( SNLS_time_pointer_type ) :: time_pointer
        TYPE ( SLLSB_pointer_type ) :: SLLSB_pointer
        TYPE ( SLLS_pointer_type ) :: SLLS_pointer
      END TYPE SNLS_pointer_type

   CONTAINS

!-*-*- S N L S _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E -*-*-

      SUBROUTINE SNLS_matlab_control_set( ps, SNLS_control, len )

!  ----------------------------------------------------------------------------

!  Set matlab control arguments from values provided to SNLS

!  Arguments

!  ps - given pointer to the structure
!  SNLS_control - SNLS control structure
!  len - length of any character component
!  nfields - only the first nfields fields in the structure will be considered

!  ----------------------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SNLS_control_type ) :: SNLS_control

!  local variables

      INTEGER :: j,  nfields
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
                                 pc, SNLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SNLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SNLS_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, SNLS_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, SNLS_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, SNLS_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, SNLS_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, SNLS_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, SNLS_control%alive_file, len )
        CASE( 'jacobian_available' )
          CALL MATLAB_get_value( ps, 'jacobian_available',                     &
                                 pc, SNLS_control%jacobian_available )
        CASE( 'subproblem_solver' )
          CALL MATLAB_get_value( ps, 'subproblem_solver',                      &
                                 pc, SNLS_control%subproblem_solver )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, SNLS_control%non_monotone )
        CASE( 'weight_update_strategy' )
          CALL MATLAB_get_value( ps, 'weight_update_strategy',                 &
                                 pc, SNLS_control%weight_update_strategy )
        CASE( 'stop_r_absolute' )
          CALL MATLAB_get_value( ps, 'stop_r_absolute',                        &
                                 pc, SNLS_control%stop_r_absolute )
        CASE( 'stop_r_relative' )
          CALL MATLAB_get_value( ps, 'stop_r_relative',                        &
                                 pc, SNLS_control%stop_r_relative )
        CASE( 'stop_pg_absolute' )
          CALL MATLAB_get_value( ps, 'stop_pg_absolute',                       &
                                 pc, SNLS_control%stop_pg_absolute )
        CASE( 'stop_pg_relative' )
          CALL MATLAB_get_value( ps, 'stop_pg_relative',                       &
                                 pc, SNLS_control%stop_pg_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, SNLS_control%stop_s )
        CASE( 'stop_pg_switch' )
          CALL MATLAB_get_value( ps, 'stop_pg_switch',                         &
                                 pc, SNLS_control%stop_pg_switch )
        CASE( 'initial_weight' )
          CALL MATLAB_get_value( ps, 'initial_weight',                         &
                                 pc, SNLS_control%initial_weight )
        CASE( 'minimum_weight' )
          CALL MATLAB_get_value( ps, 'minimum_weight',                         &
                                 pc, SNLS_control%minimum_weight )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, SNLS_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, SNLS_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, SNLS_control%eta_too_successful )
        CASE( 'weight_decrease_min' )
          CALL MATLAB_get_value( ps, 'weight_decrease_min',                    &
                                 pc, SNLS_control%weight_decrease_min )
        CASE( 'weight_decrease' )
          CALL MATLAB_get_value( ps, 'weight_decrease',                        &
                                 pc, SNLS_control%weight_decrease )
        CASE( 'weight_increase' )
          CALL MATLAB_get_value( ps, 'weight_increase',                        &
                                 pc, SNLS_control%weight_increase )
        CASE( 'weight_increase_max' )
          CALL MATLAB_get_value( ps, 'weight_increase_max',                    &
                                 pc, SNLS_control%weight_increase_max )
        CASE( 'switch_to_newton' )
          CALL MATLAB_get_value( ps, 'switch_to_newton',                       &
                                 pc, SNLS_control%switch_to_newton )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, SNLS_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, SNLS_control%clock_time_limit )
        CASE( 'newton_acceleration' )
          CALL MATLAB_get_value( ps, 'newton_acceleration',                    &
                                 pc, SNLS_control%newton_acceleration )
        CASE( 'magic_step' )
          CALL MATLAB_get_value( ps, 'magic_step',                             &
                                 pc, SNLS_control%magic_step )
        CASE( 'print_obj' )
          CALL MATLAB_get_value( ps, 'print_obj',                              &
                                 pc, SNLS_control%print_obj )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SNLS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, SNLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SNLS_control%prefix, len )
        CASE( 'SLLSB_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLLSB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLLSB_control must be a structure' )
          CALL SLLSB_matlab_control_set( pc, SNLS_control%SLLSB_control, len )
        CASE( 'SLLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLLS_control must be a structure' )
          CALL SLLS_matlab_control_set( pc, SNLS_control%SLLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SNLS_matlab_control_set

      END SUBROUTINE SNLS_matlab_control_set


!-*-*- S N L S _ M A T L A B _ C O N T R O L _ G E T   S U B R O U T I N E -*-*-

      SUBROUTINE SNLS_matlab_control_get( struct, SNLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SNLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SNLS_control - SNLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SNLS_control_type ) :: SNLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 39
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
        'error                          ', 'out                            ',  &
        'print_level                    ', 'start_print                    ',  &
        'stop_print                     ', 'print_gap                      ',  &
        'maxit                          ', 'alive_unit                     ',  &
        'alive_file                     ', 'jacobian_available             ',  &
        'subproblem_solver              ', 'non_monotone                   ',  &
        'weight_update_strategy         ', 'stop_r_absolute                ',  &
        'stop_r_relative                ', 'stop_pg_absolute               ',  &
        'stop_pg_relative               ', 'stop_s                         ',  &
        'stop_pg_switch                 ', 'initial_weight                 ',  &
        'minimum_weight                 ', 'eta_successful                 ',  &
        'eta_very_successful            ', 'eta_too_successful             ',  &
        'weight_decrease_min            ', 'weight_decrease                ',  &
        'weight_increase                ', 'weight_increase_max            ',  &
        'switch_to_newton               ', &
        'cpu_time_limit                 ', 'clock_time_limit               ',  &
        'newton_acceleration            ', 'magic_step                     ',  &
        'print_obj                      ', 'space_critical                 ',  &
        'deallocate_error_fatal         ', 'prefix                         ',  &
        'SLLS_control                   ', 'SLLSB_control                  ' /)

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
                                  SNLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SNLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SNLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  SNLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  SNLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  SNLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  SNLS_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  SNLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  SNLS_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  SNLS_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'jacobian_available',               &
                                  SNLS_control%jacobian_available )
      CALL MATLAB_fill_component( pointer, 'subproblem_solver',                &
                                  SNLS_control%subproblem_solver )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  SNLS_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'weight_update_strategy',           &
                                  SNLS_control%weight_update_strategy )
      CALL MATLAB_fill_component( pointer, 'stop_r_absolute',                  &
                                  SNLS_control%stop_r_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_r_relative',                  &
                                  SNLS_control%stop_r_relative )
      CALL MATLAB_fill_component( pointer, 'stop_pg_absolute',                 &
                                  SNLS_control%stop_pg_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_pg_relative',                 &
                                  SNLS_control%stop_pg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  SNLS_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'stop_pg_switch',                   &
                                  SNLS_control%stop_pg_switch )
      CALL MATLAB_fill_component( pointer, 'initial_weight',                   &
                                  SNLS_control%initial_weight )
      CALL MATLAB_fill_component( pointer, 'minimum_weight',                   &
                                  SNLS_control%minimum_weight )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  SNLS_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  SNLS_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  SNLS_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'weight_decrease_min',              &
                                  SNLS_control%weight_decrease_min )
      CALL MATLAB_fill_component( pointer, 'weight_decrease',                  &
                                  SNLS_control%weight_decrease )
      CALL MATLAB_fill_component( pointer, 'weight_increase',                  &
                                  SNLS_control%weight_increase )
      CALL MATLAB_fill_component( pointer, 'weight_increase_max',              &
                                  SNLS_control%weight_increase_max )
      CALL MATLAB_fill_component( pointer, 'switch_to_newton',                 &
                                  SNLS_control%switch_to_newton )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  SNLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  SNLS_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'newton_acceleration',              &
                                  SNLS_control%newton_acceleration )
      CALL MATLAB_fill_component( pointer, 'magic_step',                       &
                                  SNLS_control%magic_step )
      CALL MATLAB_fill_component( pointer, 'print_obj',                        &
                                  SNLS_control%print_obj )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SNLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  SNLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SNLS_control%prefix )

!  create the components of sub-structure SLLS_control

      CALL SLLS_matlab_control_get( pointer, SNLS_control%SLLS_control,        &
                                    'SLLS_control' )

!  create the components of sub-structure SLLSB_control

      CALL SLLSB_matlab_control_get( pointer, SNLS_control%SLLSB_control,      &
                                     'SLLSB_control' )

      RETURN

!  End of subroutine SNLS_matlab_control_get

      END SUBROUTINE SNLS_matlab_control_get


! -*- S N L S _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E -

      SUBROUTINE SNLS_matlab_inform_create( struct, SNLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SNLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SNLS_pointer - SNLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)
!
!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SNLS_pointer_type ) :: SNLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 15
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'inner_iter           ', 'r_eval               ',                   &
           'jr_eval              ', 'obj                  ',                   &
           'norm_r               ', 'norm_g               ',                   &
           'norm_pg              ', 'weight               ',                   &
           'time                 ', 'SLLS_inform          ',                   &
           'SLLSB_inform         '                              /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'slls                 ',                   &
           'sllsb                ', 'clock_total          ',                   &
           'clock_slls           ', 'clock_sllsb          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SNLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SNLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
        'status', SNLS_pointer%status )
      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
         'alloc_status', SNLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( SNLS_pointer%pointer,                 &
        'bad_alloc', SNLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
        'iter', SNLS_pointer%iter )
      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
        'inner_iter', SNLS_pointer%inner_iter )
      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
        'r_eval', SNLS_pointer%r_eval )
      CALL MATLAB_create_integer_component( SNLS_pointer%pointer,              &
        'jr_eval', SNLS_pointer%jr_eval )
      CALL MATLAB_create_real_component( SNLS_pointer%pointer,                 &
        'obj', SNLS_pointer%obj )
      CALL MATLAB_create_real_component( SNLS_pointer%pointer,                 &
        'norm_r', SNLS_pointer%norm_r )
      CALL MATLAB_create_real_component( SNLS_pointer%pointer,                 &
        'norm_g', SNLS_pointer%norm_g )
      CALL MATLAB_create_real_component( SNLS_pointer%pointer,                 &
        'norm_pg', SNLS_pointer%norm_pg )
      CALL MATLAB_create_real_component( SNLS_pointer%pointer,                 &
        'weight', SNLS_pointer%weight )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( SNLS_pointer%pointer,                   &
        'time', SNLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'total', SNLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'slls', SNLS_pointer%time_pointer%slls )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'sllsb', SNLS_pointer%time_pointer%sllsb )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'clock_total', SNLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'clock_slls', SNLS_pointer%time_pointer%clock_slls )
      CALL MATLAB_create_real_component( SNLS_pointer%time_pointer%pointer,    &
        'clock_sllsb', SNLS_pointer%time_pointer%clock_sllsb )

!  Define the components of sub-structure SLLS_inform

      CALL SLLS_matlab_inform_create( SNLS_pointer%pointer,                    &
                                      SNLS_pointer%SLLS_pointer, 'SLLS_inform' )

!  Define the components of sub-structure SLLSB_inform

      CALL SLLSB_matlab_inform_create( SNLS_pointer%pointer,                   &
                                       SNLS_pointer%SLLSB_pointer,             &
                                       'SLLSB_inform' )


      RETURN

!  End of subroutine SNLS_matlab_inform_create

      END SUBROUTINE SNLS_matlab_inform_create

!-*-*-  S N L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E  -*-*-

      SUBROUTINE SNLS_matlab_inform_get( SNLS_inform, SNLS_pointer )

!  --------------------------------------------------------------

!  Set SNLS_inform values from matlab pointers

!  Arguments

!  SNLS_inform - SNLS inform structure
!  SNLS_pointer - SNLS pointer structure

!  --------------------------------------------------------------

      TYPE ( SNLS_inform_type ) :: SNLS_inform
      TYPE ( SNLS_pointer_type ) :: SNLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SNLS_inform%status,                             &
                               mxGetPr( SNLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%alloc_status,                       &
                               mxGetPr( SNLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SNLS_pointer%pointer,                           &
                               'bad_alloc', SNLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SNLS_inform%iter,                               &
                               mxGetPr( SNLS_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%inner_iter,                         &
                               mxGetPr( SNLS_pointer%inner_iter ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%r_eval,                             &
                               mxGetPr( SNLS_pointer%r_eval ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%jr_eval,                            &
                               mxGetPr( SNLS_pointer%jr_eval ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%obj,                                &
                               mxGetPr( SNLS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%norm_r,                             &
                               mxGetPr( SNLS_pointer%norm_r ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%norm_g,                             &
                               mxGetPr( SNLS_pointer%norm_g ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%norm_pg,                            &
                               mxGetPr( SNLS_pointer%norm_pg ) )
      CALL MATLAB_copy_to_ptr( SNLS_inform%weight,                             &
                               mxGetPr( SNLS_pointer%weight ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%total, wp ),             &
                           mxGetPr( SNLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%slls, wp ),        &
                           mxGetPr( SNLS_pointer%time_pointer%slls ) )
      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%sllsb, wp ),           &
                           mxGetPr( SNLS_pointer%time_pointer%sllsb ) )
      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%clock_total, wp ),       &
                           mxGetPr( SNLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%clock_slls, wp ),  &
                          mxGetPr( SNLS_pointer%time_pointer%clock_slls ))
      CALL MATLAB_copy_to_ptr( REAL( SNLS_inform%time%clock_sllsb, wp ),     &
                           mxGetPr( SNLS_pointer%time_pointer%clock_sllsb ) )

!  projected gradient subproblem components

      CALL SLLS_matlab_inform_get( SNLS_inform%SLLS_inform,                    &
                                   SNLS_pointer%SLLS_pointer )

!  interior-point subproblemr components

      CALL SLLSB_matlab_inform_get( SNLS_inform%SLLSB_inform,                  &
                                    SNLS_pointer%SLLSB_pointer )

      RETURN

!  End of subroutine SNLS_matlab_inform_get

      END SUBROUTINE SNLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S N L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SNLS_MATLAB_TYPES
