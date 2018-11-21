#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 09/08/2018 AT 14:20 GMT.

!-*-*-*-  G A L A H A D _ L P B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.1. August 9th, 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LPB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LPB

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_LPB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPB_matlab_control_set, LPB_matlab_control_get,                &
                LPB_matlab_inform_create, LPB_matlab_inform_get

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

      TYPE, PUBLIC :: LPB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: LPB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, potential, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( LPB_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE
    CONTAINS

!-*-  L P B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE LPB_matlab_control_set( ps, LPB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LPB

!  Arguments

!  ps - given pointer to the structure
!  LPB_control - LPB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LPB_control_type ) :: LPB_control

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
                                 pc, LPB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LPB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LPB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, LPB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, LPB_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, LPB_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, LPB_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, LPB_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, LPB_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, LPB_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, LPB_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, LPB_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, LPB_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, LPB_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, LPB_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, LPB_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, LPB_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, LPB_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, LPB_control%stop_rel_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, LPB_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, LPB_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, LPB_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, LPB_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, LPB_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, LPB_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, LPB_control%reduce_infeas )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, LPB_control%obj_unbounded )
        CASE( 'potential_unbounded' )
          CALL MATLAB_get_value( ps, 'potential_unbounded',                    &
                                 pc, LPB_control%potential_unbounded )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, LPB_control%identical_bounds_tol )
        CASE( 'mu_lunge' )
          CALL MATLAB_get_value( ps, 'mu_lunge',                               &
                                 pc, LPB_control%mu_lunge )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, LPB_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, LPB_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, LPB_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, LPB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, LPB_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, LPB_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, LPB_control%treat_zero_bounds_as_general )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, LPB_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, LPB_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, LPB_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, LPB_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, LPB_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity',          &
                                 pc, LPB_control%balance_initial_complentarity )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LPB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LPB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LPB_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, LPB_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, LPB_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LPB_matlab_control_set

      END SUBROUTINE LPB_matlab_control_set

!-*-  L P B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LPB_matlab_control_get( struct, LPB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LPB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LPB_control - LPB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LPB_control_type ) :: LPB_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 48
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'maxit                          ', &
         'infeas_max                     ', 'muzero_fixed                   ', &
         'restore_problem                ', 'indicator_type                 ', &
         'arc                            ', 'series_order                   ', &
         'infinity                       ', 'stop_abs_p                     ', &
         'stop_rel_p                     ', 'stop_abs_d                     ', &
         'stop_rel_d                     ', 'stop_abs_c                     ', &
         'stop_rel_c                     ',                                    &
         'prfeas                         ', 'dufeas                         ', &
         'muzero                         ', 'tau                            ', &
         'gamma_c                        ', 'gamma_f                        ', &
         'reduce_infeas                  ', 'obj_unbounded                  ', &
         'potential_unbounded            ', 'identical_bounds_tol           ', &
         'mu_lunge                       ', 'indicator_tol_p                ', &
         'indicator_tol_pd               ', 'indicator_tol_tapia            ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'remove_dependencies            ',                                    &
         'treat_zero_bounds_as_general   ', 'just_feasible                  ', &
         'getdua                         ', 'puiseux                        ', &
         'every_order                    ', 'feasol                         ', &
         'balance_initial_complentarity  ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'FDC_control                    ', 'SBLS_control                   ' /)

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
                                  LPB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LPB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LPB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  LPB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  LPB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  LPB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  LPB_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  LPB_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  LPB_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  LPB_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  LPB_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  LPB_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  LPB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  LPB_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  LPB_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  LPB_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  LPB_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  LPB_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  LPB_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  LPB_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  LPB_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  LPB_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  LPB_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  LPB_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  LPB_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  LPB_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  LPB_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'potential_unbounded',              &
                                  LPB_control%potential_unbounded )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  LPB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_lunge',                         &
                                  LPB_control%mu_lunge )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  LPB_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  LPB_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  LPB_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  LPB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  LPB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  LPB_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  LPB_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  LPB_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  LPB_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  LPB_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  LPB_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  LPB_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  LPB_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LPB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LPB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LPB_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, LPB_control%FDC_control,           &
                                   'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, LPB_control%SBLS_control,         &
                                    'SBLS_control' )

      RETURN

!  End of subroutine LPB_matlab_control_get

      END SUBROUTINE LPB_matlab_control_get

!-*- L P B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE LPB_matlab_inform_create( struct, LPB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LPB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LPB_pointer - LPB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LPB_pointer_type ) :: LPB_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 19
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'iter                    ',             &
           'factorization_status    ', 'factorization_integer   ',             &
           'factorization_real      ', 'nfacts                  ',             &
           'nbacts                  ', 'obj                     ',             &
           'primal_infeasibility    ', 'dual_infeasibility      ',             &
           'complementary_slackness ', 'potential               ',             &
           'non_negligible_pivot    ', 'feasible                ',             &
           'time                    ', 'FDC_inform              ',             &
           'SBLS_inform             '   /)
      INTEGER * 4, PARAMETER :: t_ninform = 12
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'find_dependent       ', 'analyse              ',                   &
           'factorize            ', 'solve                ',                   &
           'clock_total          ', 'clock_preprocess     ',                   &
           'clock_find_dependent ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LPB_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LPB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'status', LPB_pointer%status )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
         'alloc_status', LPB_pointer%alloc_status )
      CALL MATLAB_create_char_component( LPB_pointer%pointer,                  &
        'bad_alloc', LPB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'iter', LPB_pointer%iter )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'factorization_status', LPB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'factorization_integer', LPB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'factorization_real', LPB_pointer%factorization_real )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'nfacts', LPB_pointer%nfacts )
      CALL MATLAB_create_integer_component( LPB_pointer%pointer,               &
        'nbacts', LPB_pointer%nbacts )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
        'obj', LPB_pointer%obj )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
         'primal_infeasibility', LPB_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
         'dual_infeasibility', LPB_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
         'complementary_slackness', LPB_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
        'potential', LPB_pointer%potential )
      CALL MATLAB_create_real_component( LPB_pointer%pointer,                  &
        'non_negligible_pivot', LPB_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( LPB_pointer%pointer,               &
        'feasible', LPB_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( LPB_pointer%pointer,                    &
        'time', LPB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'total', LPB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'preprocess', LPB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'find_dependent', LPB_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'analyse', LPB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'factorize', LPB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'solve', LPB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_total', LPB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_preprocess', LPB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_find_dependent', LPB_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_analyse', LPB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_factorize', LPB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( LPB_pointer%time_pointer%pointer,     &
        'clock_solve', LPB_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( LPB_pointer%pointer,                      &
                                     LPB_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( LPB_pointer%pointer,                     &
                                      LPB_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine LPB_matlab_inform_create

      END SUBROUTINE LPB_matlab_inform_create

!-*-*-  L P B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE LPB_matlab_inform_get( LPB_inform, LPB_pointer )

!  --------------------------------------------------------------

!  Set LPB_inform values from matlab pointers

!  Arguments

!  LPB_inform - LPB inform structure
!  LPB_pointer - LPB pointer structure

!  --------------------------------------------------------------

      TYPE ( LPB_inform_type ) :: LPB_inform
      TYPE ( LPB_pointer_type ) :: LPB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LPB_inform%status,                              &
                               mxGetPr( LPB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%alloc_status,                        &
                               mxGetPr( LPB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LPB_pointer%pointer,                            &
                               'bad_alloc', LPB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LPB_inform%iter,                                &
                               mxGetPr( LPB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%factorization_status,                &
                               mxGetPr( LPB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%factorization_integer,               &
                               mxGetPr( LPB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%factorization_real,                  &
                               mxGetPr( LPB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%nfacts,                              &
                               mxGetPr( LPB_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%nbacts,                              &
                               mxGetPr( LPB_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%obj,                                 &
                               mxGetPr( LPB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%primal_infeasibility,                &
                               mxGetPr( LPB_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%dual_infeasibility,                  &
                               mxGetPr( LPB_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%complementary_slackness,             &
                               mxGetPr( LPB_pointer%complementary_slackness ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%potential,                           &
                               mxGetPr( LPB_pointer%potential ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%non_negligible_pivot,                &
                               mxGetPr( LPB_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( LPB_inform%feasible,                            &
                               mxGetPr( LPB_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%total, wp ),              &
                               mxGetPr( LPB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%preprocess, wp ),         &
                               mxGetPr( LPB_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%find_dependent, wp ),     &
                          mxGetPr( LPB_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%analyse, wp ),            &
                               mxGetPr( LPB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%factorize, wp ),          &
                               mxGetPr( LPB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%solve, wp ),              &
                               mxGetPr( LPB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_total, wp ),        &
                      mxGetPr( LPB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( LPB_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_find_dependent, wp),&
                      mxGetPr( LPB_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_analyse, wp ),      &
                      mxGetPr( LPB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_factorize, wp ),    &
                      mxGetPr( LPB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LPB_inform%time%clock_solve, wp ),        &
                      mxGetPr( LPB_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( LPB_inform%FDC_inform,                       &
                                  LPB_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SBLS_matlab_inform_get( LPB_inform%SBLS_inform,                     &
                                   LPB_pointer%SBLS_pointer )

      RETURN

!  End of subroutine LPB_matlab_inform_get

      END SUBROUTINE LPB_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L P B _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LPB_MATLAB_TYPES
