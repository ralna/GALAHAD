#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.2 - 2023-12-18 AT 13:00 GMT.

!-*-*-*-  G A L A H A D _ C Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.2. December 18th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CLLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to CLLS

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_CLLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CLLS_matlab_control_set, CLLS_matlab_control_get,              &
                CLLS_matlab_inform_create, CLLS_matlab_inform_get

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

      TYPE, PUBLIC :: CLLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: CLLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: threads, obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( CLLS_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pounce_pointer
      END TYPE
    CONTAINS

!-*-  C Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE CLLS_matlab_control_set( ps, CLLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to CLLS

!  Arguments

!  ps - given pointer to the structure
!  CLLS_control - CLLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( CLLS_control_type ) :: CLLS_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc, mxGetField
      mwSize :: mxGetNumberOfFields
      LOGICAL :: mxIsStruct
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
!       CALL mexWarnMsgTxt( name )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, CLLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, CLLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, CLLS_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, CLLS_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, CLLS_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, CLLS_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, CLLS_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, CLLS_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, CLLS_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, CLLS_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, CLLS_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, CLLS_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, CLLS_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, CLLS_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, CLLS_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, CLLS_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, CLLS_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, CLLS_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, CLLS_control%stop_rel_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, CLLS_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, CLLS_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, CLLS_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, CLLS_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, CLLS_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, CLLS_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, CLLS_control%reduce_infeas )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, CLLS_control%identical_bounds_tol )
        CASE( 'mu_pounce' )
          CALL MATLAB_get_value( ps, 'mu_pounce',                              &
                                 pc, CLLS_control%mu_pounce )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, CLLS_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, CLLS_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, CLLS_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, CLLS_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, CLLS_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, CLLS_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, CLLS_control%treat_zero_bounds_as_general )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, CLLS_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, CLLS_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, CLLS_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, CLLS_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, CLLS_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity', pc,      &
                                 CLLS_control%balance_initial_complentarity )
        CASE( 'crossover' )
          CALL MATLAB_get_value( ps, 'crossover',                              &
                                 pc, CLLS_control%crossover )
        CASE( 'reduced_pounce_system' )
          CALL MATLAB_get_value( ps, 'reduced_pounce_system',                  &
                                 pc, CLLS_control%reduced_pounce_system )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, CLLS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, CLLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, CLLS_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, CLLS_control%FDC_control, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, CLLS_control%SLS_control, len )
        CASE( 'SLS_pounce_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_pounce_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( &
              ' component SLS_pounce_control must be a structure' )
          CALL SLS_matlab_control_set( pc, CLLS_control%SLS_pounce_control,    &
                                       len )
        END SELECT
      END DO

      RETURN

!  End of subroutine CLLS_matlab_control_set

      END SUBROUTINE CLLS_matlab_control_set

!-*-  C Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE CLLS_matlab_control_get( struct, CLLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to CLLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  CLLS_control - CLLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( CLLS_control_type ) :: CLLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 49
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
         'reduce_infeas                  ', 'identical_bounds_tol           ', &
         'mu_pounce                      ', 'indicator_tol_p                ', &
         'indicator_tol_pd               ', 'indicator_tol_tapia            ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'remove_dependencies            ',                                    &
         'treat_zero_bounds_as_general   ', 'just_feasible                  ', &
         'getdua                         ', 'puiseux                        ', &
         'every_order                    ', 'feasol                         ', &
         'balance_initial_complentarity  ', 'crossover                      ', &
         'reduced_pounce_system          ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'FDC_control                    ', 'SLS_control                    ', &
         'SLS_pounce_control             ' /)

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
                                  CLLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  CLLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  CLLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  CLLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  CLLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  CLLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  CLLS_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  CLLS_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  CLLS_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  CLLS_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  CLLS_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  CLLS_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  CLLS_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  CLLS_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  CLLS_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  CLLS_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  CLLS_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  CLLS_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  CLLS_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  CLLS_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  CLLS_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  CLLS_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  CLLS_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  CLLS_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  CLLS_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  CLLS_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  CLLS_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_pounce',                        &
                                  CLLS_control%mu_pounce )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  CLLS_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  CLLS_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  CLLS_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  CLLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  CLLS_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  CLLS_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  CLLS_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  CLLS_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  CLLS_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  CLLS_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  CLLS_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  CLLS_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  CLLS_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'crossover',                        &
                                  CLLS_control%crossover )
      CALL MATLAB_fill_component( pointer, 'reduced_pounce_system',            &
                                  CLLS_control%reduced_pounce_system )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  CLLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  CLLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  CLLS_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, CLLS_control%FDC_control,          &
                                   'FDC_control' )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, CLLS_control%SLS_control,         &
                                   'SLS_control' )

!  create the components of sub-structure SLS_pounce_control

      CALL SLS_matlab_control_get( pointer, CLLS_control%SLS_pounce_control,   &
                                   'SLS_pounce_control' )

      RETURN

!  End of subroutine CLLS_matlab_control_get

      END SUBROUTINE CLLS_matlab_control_get

!-*- C Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE CLLS_matlab_inform_create( struct, CLLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold CLLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  CLLS_pointer - CLLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( CLLS_pointer_type ) :: CLLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 20
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'iter                    ',             &
           'factorization_status    ', 'factorization_integer   ',             &
           'factorization_real      ', 'nfacts                  ',             &
           'nbacts                  ', 'threads                 ',             &
           'obj                     ', 'primal_infeasibility    ',             &
           'dual_infeasibility      ', 'complementary_slackness ',             &
           'non_negligible_pivot    ', 'feasible                ',             &
           'time                    ', 'FDC_inform              ',             &
           'SLS_inform              ', 'SLS_pounce_inform       ' /)
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
        CALL MATLAB_create_substructure( struct, name, CLLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        CLLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'status', CLLS_pointer%status )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
         'alloc_status', CLLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( CLLS_pointer%pointer,                 &
        'bad_alloc', CLLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'iter', CLLS_pointer%iter )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'factorization_status', CLLS_pointer%factorization_status )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'factorization_integer', CLLS_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'factorization_real', CLLS_pointer%factorization_real )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'nfacts', CLLS_pointer%nfacts )
      CALL MATLAB_create_integer_component( CLLS_pointer%pointer,              &
        'nbacts', CLLS_pointer%nbacts )
      CALL MATLAB_create_real_component( Clls_pointer%pointer,                 &
        'threads', Clls_pointer%threads )
      CALL MATLAB_create_real_component( CLLS_pointer%pointer,                 &
        'obj', CLLS_pointer%obj )
      CALL MATLAB_create_real_component( CLLS_pointer%pointer,                 &
         'primal_infeasibility', CLLS_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( CLLS_pointer%pointer,                 &
         'dual_infeasibility', CLLS_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( CLLS_pointer%pointer,                 &
         'complementary_slackness', CLLS_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( CLLS_pointer%pointer,                 &
        'non_negligible_pivot', CLLS_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( CLLS_pointer%pointer,              &
        'feasible', CLLS_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( CLLS_pointer%pointer,                   &
        'time', CLLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'total', CLLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'preprocess', CLLS_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'find_dependent', CLLS_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'analyse', CLLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'factorize', CLLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'solve', CLLS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_total', CLLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_preprocess', CLLS_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_find_dependent', CLLS_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_analyse', CLLS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_factorize', CLLS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( CLLS_pointer%time_pointer%pointer,    &
        'clock_solve', CLLS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( CLLS_pointer%pointer,                     &
                                     CLLS_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( CLLS_pointer%pointer,                     &
                                     CLLS_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( CLLS_pointer%pointer,                     &
                                     CLLS_pointer%SLS_pounce_pointer,          &
                                     'SLS_pounce_inform' )

      RETURN

!  End of subroutine CLLS_matlab_inform_create

      END SUBROUTINE CLLS_matlab_inform_create

!-*-*-  C Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE CLLS_matlab_inform_get( CLLS_inform, CLLS_pointer )

!  --------------------------------------------------------------

!  Set CLLS_inform values from matlab pointers

!  Arguments

!  CLLS_inform - CLLS inform structure
!  CLLS_pointer - CLLS pointer structure

!  --------------------------------------------------------------

      TYPE ( CLLS_inform_type ) :: CLLS_inform
      TYPE ( CLLS_pointer_type ) :: CLLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( CLLS_inform%status,                             &
                               mxGetPr( CLLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%alloc_status,                       &
                               mxGetPr( CLLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( CLLS_pointer%pointer,                           &
                               'bad_alloc', CLLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( CLLS_inform%iter,                               &
                               mxGetPr( CLLS_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%factorization_status,               &
                               mxGetPr( CLLS_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%factorization_integer,              &
                               mxGetPr( CLLS_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%factorization_real,                 &
                               mxGetPr( CLLS_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%nfacts,                             &
                               mxGetPr( CLLS_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%nbacts,                             &
                               mxGetPr( CLLS_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%threads,                            &
                               mxGetPr( CLLS_pointer%threads ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%obj,                                &
                               mxGetPr( CLLS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%primal_infeasibility,               &
                               mxGetPr( CLLS_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%dual_infeasibility,                 &
                               mxGetPr( CLLS_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%complementary_slackness,            &
                               mxGetPr( CLLS_pointer%complementary_slackness ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%non_negligible_pivot,               &
                               mxGetPr( CLLS_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( CLLS_inform%feasible,                           &
                               mxGetPr( CLLS_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%total, wp ),             &
                               mxGetPr( CLLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%preprocess, wp ),        &
                               mxGetPr( CLLS_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%find_dependent, wp ),    &
                          mxGetPr( CLLS_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%analyse, wp ),           &
                               mxGetPr( CLLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%factorize, wp ),         &
                               mxGetPr( CLLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%solve, wp ),             &
                               mxGetPr( CLLS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_total, wp ),       &
                      mxGetPr( CLLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_preprocess, wp ),  &
                      mxGetPr( CLLS_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_find_dependent,    &
                                     wp ),                                     &
                      mxGetPr( CLLS_pointer%time_pointer%clock_find_dependent ))
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_analyse, wp ),     &
                      mxGetPr( CLLS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_factorize, wp ),   &
                      mxGetPr( CLLS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( CLLS_inform%time%clock_solve, wp ),       &
                      mxGetPr( CLLS_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( CLLS_inform%FDC_inform,                      &
                                  CLLS_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SLS_matlab_inform_get( CLLS_inform%SLS_inform,                      &
                                  CLLS_pointer%SLS_pointer )
      CALL SLS_matlab_inform_get( CLLS_inform%SLS_pounce_inform,               &
                                  CLLS_pointer%SLS_pounce_pointer )

      RETURN

!  End of subroutine CLLS_matlab_inform_get

      END SUBROUTINE CLLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ C Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_CLLS_MATLAB_TYPES
