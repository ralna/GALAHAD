#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.5 - 2026-03-24 AT 13:20 GMT.

!-*-*-  G A L A H A D _ S L L S B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.5. March 24th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SLLSB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SLLSB

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_SLLSB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SLLSB_matlab_control_set, SLLSB_matlab_control_get,            &
                SLLSB_matlab_inform_create, SLLSB_matlab_inform_get

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

      TYPE, PUBLIC :: SLLSB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: SLLSB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: threads, obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( SLLSB_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pounce_pointer
      END TYPE
    CONTAINS

!-*-  S L L S B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE SLLSB_matlab_control_set( ps, SLLSB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SLLSB

!  Arguments

!  ps - given pointer to the structure
!  SLLSB_control - SLLSB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SLLSB_control_type ) :: SLLSB_control

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
                                 pc, SLLSB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SLLSB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SLLSB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, SLLSB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, SLLSB_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, SLLSB_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, SLLSB_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, SLLSB_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, SLLSB_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, SLLSB_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, SLLSB_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, SLLSB_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, SLLSB_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, SLLSB_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, SLLSB_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, SLLSB_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, SLLSB_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, SLLSB_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, SLLSB_control%stop_rel_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, SLLSB_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, SLLSB_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, SLLSB_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, SLLSB_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, SLLSB_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, SLLSB_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, SLLSB_control%reduce_infeas )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, SLLSB_control%identical_bounds_tol )
        CASE( 'mu_pounce' )
          CALL MATLAB_get_value( ps, 'mu_pounce',                              &
                                 pc, SLLSB_control%mu_pounce )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, SLLSB_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, SLLSB_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, SLLSB_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, SLLSB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, SLLSB_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, SLLSB_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, SLLSB_control%treat_zero_bounds_as_general)
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, SLLSB_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, SLLSB_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, SLLSB_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, SLLSB_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, SLLSB_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity', pc,      &
                                 SLLSB_control%balance_initial_complentarity )
        CASE( 'crossover' )
          CALL MATLAB_get_value( ps, 'crossover',                              &
                                 pc, SLLSB_control%crossover )
        CASE( 'reduced_pounce_system' )
          CALL MATLAB_get_value( ps, 'reduced_pounce_system',                  &
                                 pc, SLLSB_control%reduced_pounce_system )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SLLSB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, SLLSB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SLLSB_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, SLLSB_control%FDC_control, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, SLLSB_control%SLS_control, len )
        CASE( 'SLS_pounce_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_pounce_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( &
              ' component SLS_pounce_control must be a structure' )
          CALL SLS_matlab_control_set( pc, SLLSB_control%SLS_pounce_control,   &
                                       len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SLLSB_matlab_control_set

      END SUBROUTINE SLLSB_matlab_control_set

!-*-  S L L S B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE SLLSB_matlab_control_get( struct, SLLSB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SLLSB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SLLSB_control - SLLSB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLLSB_control_type ) :: SLLSB_control
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
                                  SLLSB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SLLSB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SLLSB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  SLLSB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  SLLSB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  SLLSB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  SLLSB_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  SLLSB_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  SLLSB_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  SLLSB_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  SLLSB_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  SLLSB_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  SLLSB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  SLLSB_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  SLLSB_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  SLLSB_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  SLLSB_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  SLLSB_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  SLLSB_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  SLLSB_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  SLLSB_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  SLLSB_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  SLLSB_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  SLLSB_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  SLLSB_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  SLLSB_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  SLLSB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_pounce',                        &
                                  SLLSB_control%mu_pounce )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  SLLSB_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  SLLSB_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  SLLSB_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  SLLSB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  SLLSB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  SLLSB_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  SLLSB_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  SLLSB_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  SLLSB_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  SLLSB_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  SLLSB_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  SLLSB_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  SLLSB_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'crossover',                        &
                                  SLLSB_control%crossover )
      CALL MATLAB_fill_component( pointer, 'reduced_pounce_system',            &
                                  SLLSB_control%reduced_pounce_system )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SLLSB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  SLLSB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SLLSB_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, SLLSB_control%FDC_control,         &
                                   'FDC_control' )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, SLLSB_control%SLS_control,         &
                                   'SLS_control' )

!  create the components of sub-structure SLS_pounce_control

      CALL SLS_matlab_control_get( pointer, SLLSB_control%SLS_pounce_control,  &
                                   'SLS_pounce_control' )

      RETURN

!  End of subroutine SLLSB_matlab_control_get

      END SUBROUTINE SLLSB_matlab_control_get

!-  S L L S B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -

      SUBROUTINE SLLSB_matlab_inform_create( struct, SLLSB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SLLSB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SLLSB_pointer - SLLSB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLLSB_pointer_type ) :: SLLSB_pointer
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
        CALL MATLAB_create_substructure( struct, name, SLLSB_pointer%pointer,  &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SLLSB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'status', SLLSB_pointer%status )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
         'alloc_status', SLLSB_pointer%alloc_status )
      CALL MATLAB_create_char_component( SLLSB_pointer%pointer,                &
        'bad_alloc', SLLSB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'iter', SLLSB_pointer%iter )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'factorization_status', SLLSB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'factorization_integer', SLLSB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'factorization_real', SLLSB_pointer%factorization_real )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'nfacts', SLLSB_pointer%nfacts )
      CALL MATLAB_create_integer_component( SLLSB_pointer%pointer,             &
        'nbacts', SLLSB_pointer%nbacts )
      CALL MATLAB_create_real_component( Sllsb_pointer%pointer,                &
        'threads', Sllsb_pointer%threads )
      CALL MATLAB_create_real_component( SLLSB_pointer%pointer,                &
        'obj', SLLSB_pointer%obj )
      CALL MATLAB_create_real_component( SLLSB_pointer%pointer,                &
         'primal_infeasibility', SLLSB_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( SLLSB_pointer%pointer,                &
         'dual_infeasibility', SLLSB_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( SLLSB_pointer%pointer,                &
         'complementary_slackness', SLLSB_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( SLLSB_pointer%pointer,                &
        'non_negligible_pivot', SLLSB_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( SLLSB_pointer%pointer,             &
        'feasible', SLLSB_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( SLLSB_pointer%pointer,                  &
        'time', SLLSB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'total', SLLSB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'preprocess', SLLSB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'find_dependent', SLLSB_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'analyse', SLLSB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'factorize', SLLSB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'solve', SLLSB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_total', SLLSB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_preprocess', SLLSB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_find_dependent', SLLSB_pointer%time_pointer%clock_find_dependent)
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_analyse', SLLSB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_factorize', SLLSB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( SLLSB_pointer%time_pointer%pointer,   &
        'clock_solve', SLLSB_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( SLLSB_pointer%pointer,                    &
                                     SLLSB_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( SLLSB_pointer%pointer,                    &
                                     SLLSB_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( SLLSB_pointer%pointer,                    &
                                     SLLSB_pointer%SLS_pounce_pointer,         &
                                     'SLS_pounce_inform' )

      RETURN

!  End of subroutine SLLSB_matlab_inform_create

      END SUBROUTINE SLLSB_matlab_inform_create

!-*-  S L L S B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE SLLSB_matlab_inform_get( SLLSB_inform, SLLSB_pointer )

!  --------------------------------------------------------------

!  Set SLLSB_inform values from matlab pointers

!  Arguments

!  SLLSB_inform - SLLSB inform structure
!  SLLSB_pointer - SLLSB pointer structure

!  --------------------------------------------------------------

      TYPE ( SLLSB_inform_type ) :: SLLSB_inform
      TYPE ( SLLSB_pointer_type ) :: SLLSB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SLLSB_inform%status,                            &
                               mxGetPr( SLLSB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%alloc_status,                      &
                               mxGetPr( SLLSB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SLLSB_pointer%pointer,                          &
                               'bad_alloc', SLLSB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%iter,                              &
                               mxGetPr( SLLSB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%factorization_status,              &
                               mxGetPr( SLLSB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%factorization_integer,             &
                               mxGetPr( SLLSB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%factorization_real,                &
                               mxGetPr( SLLSB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%nfacts,                            &
                               mxGetPr( SLLSB_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%nbacts,                            &
                               mxGetPr( SLLSB_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%threads,                           &
                               mxGetPr( SLLSB_pointer%threads ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%obj,                               &
                               mxGetPr( SLLSB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%primal_infeasibility,              &
                               mxGetPr( SLLSB_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%dual_infeasibility,                &
                               mxGetPr( SLLSB_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%complementary_slackness,           &
                               mxGetPr( SLLSB_pointer%complementary_slackness ))
      CALL MATLAB_copy_to_ptr( SLLSB_inform%non_negligible_pivot,              &
                               mxGetPr( SLLSB_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( SLLSB_inform%feasible,                          &
                               mxGetPr( SLLSB_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%total, wp ),            &
                               mxGetPr( SLLSB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%preprocess, wp ),       &
                               mxGetPr( SLLSB_pointer%time_pointer%preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%find_dependent, wp ),   &
                          mxGetPr( SLLSB_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%analyse, wp ),          &
                               mxGetPr( SLLSB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%factorize, wp ),        &
                               mxGetPr( SLLSB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%solve, wp ),            &
                               mxGetPr( SLLSB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_total, wp ),      &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_preprocess, wp ), &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_find_dependent,   &
                                     wp ),                                     &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_find_dependent))
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_analyse, wp ),    &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_factorize, wp ),  &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLSB_inform%time%clock_solve, wp ),      &
                      mxGetPr( SLLSB_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( SLLSB_inform%FDC_inform,                     &
                                  SLLSB_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SLS_matlab_inform_get( SLLSB_inform%SLS_inform,                     &
                                  SLLSB_pointer%SLS_pointer )
      CALL SLS_matlab_inform_get( SLLSB_inform%SLS_pounce_inform,              &
                                  SLLSB_pointer%SLS_pounce_pointer )

      RETURN

!  End of subroutine SLLSB_matlab_inform_get

      END SUBROUTINE SLLSB_matlab_inform_get

!-*-*-  E N D  o f  G A L A H A D _ B L L S B _ T Y P E S   M O D U L E  -*-*-

    END MODULE GALAHAD_SLLSB_MATLAB_TYPES
