#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-05-21 AT 10:50 GMT.

!-*-*-*-  G A L A H A D _ B N L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.5. May 21st, 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BNLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BNLS

      USE GALAHAD_MATLAB
      USE GALAHAD_BLLS_MATLAB_TYPES
      USE GALAHAD_BLLSB_MATLAB_TYPES
      USE GALAHAD_BNLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BNLS_matlab_control_set, BNLS_matlab_control_get,              &
                BNLS_matlab_inform_create, BNLS_matlab_inform_get

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

      TYPE, PUBLIC :: BNLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, blls, bllsb
        mwPointer :: clock_total, clock_blls, clock_bllsb
      END TYPE BNLS_time_pointer_type

      TYPE, PUBLIC :: BNLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, inner_iter, r_eval, jr_eval
        mwPointer :: obj, norm_r, norm_g, norm_pg, weight
        mwPointer :: time
        TYPE ( BNLS_time_pointer_type ) :: time_pointer
        TYPE ( BLLSB_pointer_type ) :: BLLSB_pointer
        TYPE ( BLLS_pointer_type ) :: BLLS_pointer
      END TYPE BNLS_pointer_type

   CONTAINS

!-*-*- B N L S _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E -*-*-

      SUBROUTINE BNLS_matlab_control_set( ps, BNLS_control, len )

!  ----------------------------------------------------------------------------

!  Set matlab control arguments from values provided to BNLS

!  Arguments

!  ps - given pointer to the structure
!  BNLS_control - BNLS control structure
!  len - length of any character component
!  nfields - only the first nfields fields in the structure will be considered

!  ----------------------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BNLS_control_type ) :: BNLS_control

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
                                 pc, BNLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BNLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BNLS_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BNLS_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BNLS_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, BNLS_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, BNLS_control%maxit )
        CASE( 'alive_unit' )
          CALL MATLAB_get_value( ps, 'alive_unit',                             &
                                 pc, BNLS_control%alive_unit )
        CASE( 'alive_file' )
          CALL galmxGetCharacter( ps, 'alive_file',                            &
                                  pc, BNLS_control%alive_file, len )
        CASE( 'jacobian_available' )
          CALL MATLAB_get_value( ps, 'jacobian_available',                     &
                                 pc, BNLS_control%jacobian_available )
        CASE( 'subproblem_solver' )
          CALL MATLAB_get_value( ps, 'subproblem_solver',                      &
                                 pc, BNLS_control%subproblem_solver )
        CASE( 'non_monotone' )
          CALL MATLAB_get_value( ps, 'non_monotone',                           &
                                 pc, BNLS_control%non_monotone )
        CASE( 'weight_update_strategy' )
          CALL MATLAB_get_value( ps, 'weight_update_strategy',                 &
                                 pc, BNLS_control%weight_update_strategy )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BNLS_control%infinity )
        CASE( 'stop_r_absolute' )
          CALL MATLAB_get_value( ps, 'stop_r_absolute',                        &
                                 pc, BNLS_control%stop_r_absolute )
        CASE( 'stop_r_relative' )
          CALL MATLAB_get_value( ps, 'stop_r_relative',                        &
                                 pc, BNLS_control%stop_r_relative )
        CASE( 'stop_pg_absolute' )
          CALL MATLAB_get_value( ps, 'stop_pg_absolute',                       &
                                 pc, BNLS_control%stop_pg_absolute )
        CASE( 'stop_pg_relative' )
          CALL MATLAB_get_value( ps, 'stop_pg_relative',                       &
                                 pc, BNLS_control%stop_pg_relative )
        CASE( 'stop_s' )
          CALL MATLAB_get_value( ps, 'stop_s',                                 &
                                 pc, BNLS_control%stop_s )
        CASE( 'stop_pg_switch' )
          CALL MATLAB_get_value( ps, 'stop_pg_switch',                         &
                                 pc, BNLS_control%stop_pg_switch )
        CASE( 'initial_weight' )
          CALL MATLAB_get_value( ps, 'initial_weight',                         &
                                 pc, BNLS_control%initial_weight )
        CASE( 'minimum_weight' )
          CALL MATLAB_get_value( ps, 'minimum_weight',                         &
                                 pc, BNLS_control%minimum_weight )
        CASE( 'eta_successful' )
          CALL MATLAB_get_value( ps, 'eta_successful',                         &
                                 pc, BNLS_control%eta_successful )
        CASE( 'eta_very_successful' )
          CALL MATLAB_get_value( ps, 'eta_very_successful',                    &
                                 pc, BNLS_control%eta_very_successful )
        CASE( 'eta_too_successful' )
          CALL MATLAB_get_value( ps, 'eta_too_successful',                     &
                                 pc, BNLS_control%eta_too_successful )
        CASE( 'weight_decrease_min' )
          CALL MATLAB_get_value( ps, 'weight_decrease_min',                    &
                                 pc, BNLS_control%weight_decrease_min )
        CASE( 'weight_decrease' )
          CALL MATLAB_get_value( ps, 'weight_decrease',                        &
                                 pc, BNLS_control%weight_decrease )
        CASE( 'weight_increase' )
          CALL MATLAB_get_value( ps, 'weight_increase',                        &
                                 pc, BNLS_control%weight_increase )
        CASE( 'weight_increase_max' )
          CALL MATLAB_get_value( ps, 'weight_increase_max',                    &
                                 pc, BNLS_control%weight_increase_max )
        CASE( 'switch_to_newton' )
          CALL MATLAB_get_value( ps, 'switch_to_newton',                       &
                                 pc, BNLS_control%switch_to_newton )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BNLS_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, BNLS_control%clock_time_limit )
        CASE( 'newton_acceleration' )
          CALL MATLAB_get_value( ps, 'newton_acceleration',                    &
                                 pc, BNLS_control%newton_acceleration )
        CASE( 'magic_step' )
          CALL MATLAB_get_value( ps, 'magic_step',                             &
                                 pc, BNLS_control%magic_step )
        CASE( 'print_obj' )
          CALL MATLAB_get_value( ps, 'print_obj',                              &
                                 pc, BNLS_control%print_obj )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BNLS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BNLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BNLS_control%prefix, len )
        CASE( 'BLLSB_control' )
          pc = mxGetField( ps, 1_mwi_, 'BLLSB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component BLLSB_control must be a structure' )
          CALL BLLSB_matlab_control_set( pc, BNLS_control%BLLSB_control, len )
        CASE( 'BLLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'BLLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component BLLS_control must be a structure' )
          CALL BLLS_matlab_control_set( pc, BNLS_control%BLLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BNLS_matlab_control_set

      END SUBROUTINE BNLS_matlab_control_set


!-*-*- B N L S _ M A T L A B _ C O N T R O L _ G E T   S U B R O U T I N E -*-*-

      SUBROUTINE BNLS_matlab_control_get( struct, BNLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BNLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BNLS_control - BNLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BNLS_control_type ) :: BNLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 40
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
        'error                          ', 'out                            ',  &
        'print_level                    ', 'start_print                    ',  &
        'stop_print                     ', 'print_gap                      ',  &
        'maxit                          ', 'alive_unit                     ',  &
        'alive_file                     ', 'jacobian_available             ',  &
        'subproblem_solver              ', 'non_monotone                   ',  &
        'weight_update_strategy         ', 'infinity                       ',  &
        'stop_r_absolute                ',  &
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
        'BLLS_control                   ', 'BLLSB_control                  ' /)

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
                                  BNLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BNLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BNLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BNLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BNLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BNLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  BNLS_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  BNLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'alive_unit',                       &
                                  BNLS_control%alive_unit )
      CALL MATLAB_fill_component( pointer, 'alive_file',                       &
                                  BNLS_control%alive_file )
      CALL MATLAB_fill_component( pointer, 'jacobian_available',               &
                                  BNLS_control%jacobian_available )
      CALL MATLAB_fill_component( pointer, 'subproblem_solver',                &
                                  BNLS_control%subproblem_solver )
      CALL MATLAB_fill_component( pointer, 'non_monotone',                     &
                                  BNLS_control%non_monotone )
      CALL MATLAB_fill_component( pointer, 'weight_update_strategy',           &
                                  BNLS_control%weight_update_strategy )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BNLS_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_r_absolute',                  &
                                  BNLS_control%stop_r_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_r_relative',                  &
                                  BNLS_control%stop_r_relative )
      CALL MATLAB_fill_component( pointer, 'stop_pg_absolute',                 &
                                  BNLS_control%stop_pg_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_pg_relative',                 &
                                  BNLS_control%stop_pg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_s',                           &
                                  BNLS_control%stop_s )
      CALL MATLAB_fill_component( pointer, 'stop_pg_switch',                   &
                                  BNLS_control%stop_pg_switch )
      CALL MATLAB_fill_component( pointer, 'initial_weight',                   &
                                  BNLS_control%initial_weight )
      CALL MATLAB_fill_component( pointer, 'minimum_weight',                   &
                                  BNLS_control%minimum_weight )
      CALL MATLAB_fill_component( pointer, 'eta_successful',                   &
                                  BNLS_control%eta_successful )
      CALL MATLAB_fill_component( pointer, 'eta_very_successful',              &
                                  BNLS_control%eta_very_successful )
      CALL MATLAB_fill_component( pointer, 'eta_too_successful',               &
                                  BNLS_control%eta_too_successful )
      CALL MATLAB_fill_component( pointer, 'weight_decrease_min',              &
                                  BNLS_control%weight_decrease_min )
      CALL MATLAB_fill_component( pointer, 'weight_decrease',                  &
                                  BNLS_control%weight_decrease )
      CALL MATLAB_fill_component( pointer, 'weight_increase',                  &
                                  BNLS_control%weight_increase )
      CALL MATLAB_fill_component( pointer, 'weight_increase_max',              &
                                  BNLS_control%weight_increase_max )
      CALL MATLAB_fill_component( pointer, 'switch_to_newton',                 &
                                  BNLS_control%switch_to_newton )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BNLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  BNLS_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'newton_acceleration',              &
                                  BNLS_control%newton_acceleration )
      CALL MATLAB_fill_component( pointer, 'magic_step',                       &
                                  BNLS_control%magic_step )
      CALL MATLAB_fill_component( pointer, 'print_obj',                        &
                                  BNLS_control%print_obj )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BNLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BNLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BNLS_control%prefix )

!  create the components of sub-structure BLLS_control

      CALL BLLS_matlab_control_get( pointer, BNLS_control%BLLS_control,        &
                                    'BLLS_control' )

!  create the components of sub-structure BLLSB_control

      CALL BLLSB_matlab_control_get( pointer, BNLS_control%BLLSB_control,      &
                                     'BLLSB_control' )

      RETURN

!  End of subroutine BNLS_matlab_control_get

      END SUBROUTINE BNLS_matlab_control_get


! -*- B N L S _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E -

      SUBROUTINE BNLS_matlab_inform_create( struct, BNLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BNLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BNLS_pointer - BNLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)
!
!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BNLS_pointer_type ) :: BNLS_pointer
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
           'time                 ', 'BLLS_inform          ',                   &
           'BLLSB_inform         '                              /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'blls                 ',                   &
           'bllsb                ', 'clock_total          ',                   &
           'clock_blls           ', 'clock_bllsb          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, BNLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BNLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
        'status', BNLS_pointer%status )
      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
         'alloc_status', BNLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( BNLS_pointer%pointer,                 &
        'bad_alloc', BNLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
        'iter', BNLS_pointer%iter )
      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
        'inner_iter', BNLS_pointer%inner_iter )
      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
        'r_eval', BNLS_pointer%r_eval )
      CALL MATLAB_create_integer_component( BNLS_pointer%pointer,              &
        'jr_eval', BNLS_pointer%jr_eval )
      CALL MATLAB_create_real_component( BNLS_pointer%pointer,                 &
        'obj', BNLS_pointer%obj )
      CALL MATLAB_create_real_component( BNLS_pointer%pointer,                 &
        'norm_r', BNLS_pointer%norm_r )
      CALL MATLAB_create_real_component( BNLS_pointer%pointer,                 &
        'norm_g', BNLS_pointer%norm_g )
      CALL MATLAB_create_real_component( BNLS_pointer%pointer,                 &
        'norm_pg', BNLS_pointer%norm_pg )
      CALL MATLAB_create_real_component( BNLS_pointer%pointer,                 &
        'weight', BNLS_pointer%weight )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BNLS_pointer%pointer,                   &
        'time', BNLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'total', BNLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'blls', BNLS_pointer%time_pointer%blls )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'bllsb', BNLS_pointer%time_pointer%bllsb )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'clock_total', BNLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'clock_blls', BNLS_pointer%time_pointer%clock_blls )
      CALL MATLAB_create_real_component( BNLS_pointer%time_pointer%pointer,    &
        'clock_bllsb', BNLS_pointer%time_pointer%clock_bllsb )

!  Define the components of sub-structure BLLS_inform

      CALL BLLS_matlab_inform_create( BNLS_pointer%pointer,                    &
                                      BNLS_pointer%BLLS_pointer, 'BLLS_inform' )

!  Define the components of sub-structure BLLSB_inform

      CALL BLLSB_matlab_inform_create( BNLS_pointer%pointer,                   &
                                       BNLS_pointer%BLLSB_pointer,             &
                                       'BLLSB_inform' )


      RETURN

!  End of subroutine BNLS_matlab_inform_create

      END SUBROUTINE BNLS_matlab_inform_create

!-*-*-  B N L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E  -*-*-

      SUBROUTINE BNLS_matlab_inform_get( BNLS_inform, BNLS_pointer )

!  --------------------------------------------------------------

!  Set BNLS_inform values from matlab pointers

!  Arguments

!  BNLS_inform - BNLS inform structure
!  BNLS_pointer - BNLS pointer structure

!  --------------------------------------------------------------

      TYPE ( BNLS_inform_type ) :: BNLS_inform
      TYPE ( BNLS_pointer_type ) :: BNLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BNLS_inform%status,                             &
                               mxGetPr( BNLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%alloc_status,                       &
                               mxGetPr( BNLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BNLS_pointer%pointer,                           &
                               'bad_alloc', BNLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BNLS_inform%iter,                               &
                               mxGetPr( BNLS_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%inner_iter,                         &
                               mxGetPr( BNLS_pointer%inner_iter ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%r_eval,                             &
                               mxGetPr( BNLS_pointer%r_eval ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%jr_eval,                            &
                               mxGetPr( BNLS_pointer%jr_eval ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%obj,                                &
                               mxGetPr( BNLS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%norm_r,                             &
                               mxGetPr( BNLS_pointer%norm_r ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%norm_g,                             &
                               mxGetPr( BNLS_pointer%norm_g ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%norm_pg,                            &
                               mxGetPr( BNLS_pointer%norm_pg ) )
      CALL MATLAB_copy_to_ptr( BNLS_inform%weight,                             &
                               mxGetPr( BNLS_pointer%weight ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%total, wp ),             &
                           mxGetPr( BNLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%blls, wp ),        &
                           mxGetPr( BNLS_pointer%time_pointer%blls ) )
      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%bllsb, wp ),           &
                           mxGetPr( BNLS_pointer%time_pointer%bllsb ) )
      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%clock_total, wp ),       &
                           mxGetPr( BNLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%clock_blls, wp ),  &
                          mxGetPr( BNLS_pointer%time_pointer%clock_blls ))
      CALL MATLAB_copy_to_ptr( REAL( BNLS_inform%time%clock_bllsb, wp ),     &
                           mxGetPr( BNLS_pointer%time_pointer%clock_bllsb ) )

!  projected gradient subproblem components

      CALL BLLS_matlab_inform_get( BNLS_inform%BLLS_inform,                    &
                                   BNLS_pointer%BLLS_pointer )

!  interior-point subproblemr components

      CALL BLLSB_matlab_inform_get( BNLS_inform%BLLSB_inform,                  &
                                    BNLS_pointer%BLLSB_pointer )

      RETURN

!  End of subroutine BNLS_matlab_inform_get

      END SUBROUTINE BNLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B N L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_BNLS_MATLAB_TYPES
