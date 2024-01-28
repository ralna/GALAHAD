#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.3 - 2023-12-27 AT 13:00 GMT.

!-*-*-  G A L A H A D _ B L L S B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.3. December 27th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BLLSB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BLLSB

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_BLLSB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BLLSB_matlab_control_set, BLLSB_matlab_control_get,            &
                BLLSB_matlab_inform_create, BLLSB_matlab_inform_get

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

      TYPE, PUBLIC :: BLLSB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: BLLSB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: threads, obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( BLLSB_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pounce_pointer
      END TYPE
    CONTAINS

!-*-  C Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BLLSB_matlab_control_set( ps, BLLSB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BLLSB

!  Arguments

!  ps - given pointer to the structure
!  BLLSB_control - BLLSB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BLLSB_control_type ) :: BLLSB_control

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
                                 pc, BLLSB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BLLSB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BLLSB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BLLSB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BLLSB_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, BLLSB_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, BLLSB_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, BLLSB_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, BLLSB_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, BLLSB_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, BLLSB_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, BLLSB_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BLLSB_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, BLLSB_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, BLLSB_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, BLLSB_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, BLLSB_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, BLLSB_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, BLLSB_control%stop_rel_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, BLLSB_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, BLLSB_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, BLLSB_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, BLLSB_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, BLLSB_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, BLLSB_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, BLLSB_control%reduce_infeas )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, BLLSB_control%identical_bounds_tol )
        CASE( 'mu_pounce' )
          CALL MATLAB_get_value( ps, 'mu_pounce',                              &
                                 pc, BLLSB_control%mu_pounce )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, BLLSB_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, BLLSB_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, BLLSB_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BLLSB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, BLLSB_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, BLLSB_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, BLLSB_control%treat_zero_bounds_as_general)
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, BLLSB_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, BLLSB_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, BLLSB_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, BLLSB_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, BLLSB_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity', pc,      &
                                 BLLSB_control%balance_initial_complentarity )
        CASE( 'crossover' )
          CALL MATLAB_get_value( ps, 'crossover',                              &
                                 pc, BLLSB_control%crossover )
        CASE( 'reduced_pounce_system' )
          CALL MATLAB_get_value( ps, 'reduced_pounce_system',                  &
                                 pc, BLLSB_control%reduced_pounce_system )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BLLSB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BLLSB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BLLSB_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, BLLSB_control%FDC_control, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, BLLSB_control%SLS_control, len )
        CASE( 'SLS_pounce_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_pounce_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( &
              ' component SLS_pounce_control must be a structure' )
          CALL SLS_matlab_control_set( pc, BLLSB_control%SLS_pounce_control,   &
                                       len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BLLSB_matlab_control_set

      END SUBROUTINE BLLSB_matlab_control_set

!-*-  C Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BLLSB_matlab_control_get( struct, BLLSB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BLLSB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BLLSB_control - BLLSB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BLLSB_control_type ) :: BLLSB_control
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
                                  BLLSB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BLLSB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BLLSB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BLLSB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BLLSB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  BLLSB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  BLLSB_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  BLLSB_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  BLLSB_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  BLLSB_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  BLLSB_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  BLLSB_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BLLSB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  BLLSB_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  BLLSB_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  BLLSB_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  BLLSB_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  BLLSB_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  BLLSB_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  BLLSB_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  BLLSB_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  BLLSB_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  BLLSB_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  BLLSB_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  BLLSB_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  BLLSB_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  BLLSB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_pounce',                        &
                                  BLLSB_control%mu_pounce )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  BLLSB_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  BLLSB_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  BLLSB_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BLLSB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  BLLSB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  BLLSB_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  BLLSB_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  BLLSB_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  BLLSB_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  BLLSB_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  BLLSB_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  BLLSB_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  BLLSB_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'crossover',                        &
                                  BLLSB_control%crossover )
      CALL MATLAB_fill_component( pointer, 'reduced_pounce_system',            &
                                  BLLSB_control%reduced_pounce_system )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BLLSB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BLLSB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BLLSB_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, BLLSB_control%FDC_control,         &
                                   'FDC_control' )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, BLLSB_control%SLS_control,         &
                                   'SLS_control' )

!  create the components of sub-structure SLS_pounce_control

      CALL SLS_matlab_control_get( pointer, BLLSB_control%SLS_pounce_control,  &
                                   'SLS_pounce_control' )

      RETURN

!  End of subroutine BLLSB_matlab_control_get

      END SUBROUTINE BLLSB_matlab_control_get

!-*- C Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE BLLSB_matlab_inform_create( struct, BLLSB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BLLSB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BLLSB_pointer - BLLSB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BLLSB_pointer_type ) :: BLLSB_pointer
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
        CALL MATLAB_create_substructure( struct, name, BLLSB_pointer%pointer,  &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BLLSB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'status', BLLSB_pointer%status )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
         'alloc_status', BLLSB_pointer%alloc_status )
      CALL MATLAB_create_char_component( BLLSB_pointer%pointer,                &
        'bad_alloc', BLLSB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'iter', BLLSB_pointer%iter )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'factorization_status', BLLSB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'factorization_integer', BLLSB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'factorization_real', BLLSB_pointer%factorization_real )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'nfacts', BLLSB_pointer%nfacts )
      CALL MATLAB_create_integer_component( BLLSB_pointer%pointer,             &
        'nbacts', BLLSB_pointer%nbacts )
      CALL MATLAB_create_real_component( Bllsb_pointer%pointer,                &
        'threads', Bllsb_pointer%threads )
      CALL MATLAB_create_real_component( BLLSB_pointer%pointer,                &
        'obj', BLLSB_pointer%obj )
      CALL MATLAB_create_real_component( BLLSB_pointer%pointer,                &
         'primal_infeasibility', BLLSB_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( BLLSB_pointer%pointer,                &
         'dual_infeasibility', BLLSB_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( BLLSB_pointer%pointer,                &
         'complementary_slackness', BLLSB_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( BLLSB_pointer%pointer,                &
        'non_negligible_pivot', BLLSB_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( BLLSB_pointer%pointer,             &
        'feasible', BLLSB_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BLLSB_pointer%pointer,                  &
        'time', BLLSB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'total', BLLSB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'preprocess', BLLSB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'find_dependent', BLLSB_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'analyse', BLLSB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'factorize', BLLSB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'solve', BLLSB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_total', BLLSB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_preprocess', BLLSB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_find_dependent', BLLSB_pointer%time_pointer%clock_find_dependent)
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_analyse', BLLSB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_factorize', BLLSB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( BLLSB_pointer%time_pointer%pointer,   &
        'clock_solve', BLLSB_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( BLLSB_pointer%pointer,                    &
                                     BLLSB_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( BLLSB_pointer%pointer,                    &
                                     BLLSB_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( BLLSB_pointer%pointer,                    &
                                     BLLSB_pointer%SLS_pounce_pointer,         &
                                     'SLS_pounce_inform' )

      RETURN

!  End of subroutine BLLSB_matlab_inform_create

      END SUBROUTINE BLLSB_matlab_inform_create

!-*-*-  C Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE BLLSB_matlab_inform_get( BLLSB_inform, BLLSB_pointer )

!  --------------------------------------------------------------

!  Set BLLSB_inform values from matlab pointers

!  Arguments

!  BLLSB_inform - BLLSB inform structure
!  BLLSB_pointer - BLLSB pointer structure

!  --------------------------------------------------------------

      TYPE ( BLLSB_inform_type ) :: BLLSB_inform
      TYPE ( BLLSB_pointer_type ) :: BLLSB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BLLSB_inform%status,                            &
                               mxGetPr( BLLSB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%alloc_status,                      &
                               mxGetPr( BLLSB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BLLSB_pointer%pointer,                          &
                               'bad_alloc', BLLSB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%iter,                              &
                               mxGetPr( BLLSB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%factorization_status,              &
                               mxGetPr( BLLSB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%factorization_integer,             &
                               mxGetPr( BLLSB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%factorization_real,                &
                               mxGetPr( BLLSB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%nfacts,                            &
                               mxGetPr( BLLSB_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%nbacts,                            &
                               mxGetPr( BLLSB_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%threads,                           &
                               mxGetPr( BLLSB_pointer%threads ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%obj,                               &
                               mxGetPr( BLLSB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%primal_infeasibility,              &
                               mxGetPr( BLLSB_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%dual_infeasibility,                &
                               mxGetPr( BLLSB_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%complementary_slackness,           &
                               mxGetPr( BLLSB_pointer%complementary_slackness ))
      CALL MATLAB_copy_to_ptr( BLLSB_inform%non_negligible_pivot,              &
                               mxGetPr( BLLSB_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( BLLSB_inform%feasible,                          &
                               mxGetPr( BLLSB_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%total, wp ),            &
                               mxGetPr( BLLSB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%preprocess, wp ),       &
                               mxGetPr( BLLSB_pointer%time_pointer%preprocess ))
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%find_dependent, wp ),   &
                          mxGetPr( BLLSB_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%analyse, wp ),          &
                               mxGetPr( BLLSB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%factorize, wp ),        &
                               mxGetPr( BLLSB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%solve, wp ),            &
                               mxGetPr( BLLSB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_total, wp ),      &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_preprocess, wp ), &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_find_dependent,   &
                                     wp ),                                     &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_find_dependent))
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_analyse, wp ),    &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_factorize, wp ),  &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLSB_inform%time%clock_solve, wp ),      &
                      mxGetPr( BLLSB_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( BLLSB_inform%FDC_inform,                     &
                                  BLLSB_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SLS_matlab_inform_get( BLLSB_inform%SLS_inform,                     &
                                  BLLSB_pointer%SLS_pointer )
      CALL SLS_matlab_inform_get( BLLSB_inform%SLS_pounce_inform,              &
                                  BLLSB_pointer%SLS_pounce_pointer )

      RETURN

!  End of subroutine BLLSB_matlab_inform_get

      END SUBROUTINE BLLSB_matlab_inform_get

!-*-*-  E N D  o f  G A L A H A D _ B L L S B _ T Y P E S   M O D U L E  -*-*-

    END MODULE GALAHAD_BLLSB_MATLAB_TYPES
