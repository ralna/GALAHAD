#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.2 - 2023-12-21 AT 10:30 GMT.

!-*-*-  G A L A H A D _  B Q P B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BQPB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BQPB

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_BQPB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BQPB_matlab_control_set, BQPB_matlab_control_get,              &
                BQPB_matlab_inform_create, BQPB_matlab_inform_get

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

      TYPE, PUBLIC :: BQPB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: BQPB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, potential, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( BQPB_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE
    CONTAINS

!-*-   B Q P B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BQPB_matlab_control_set( ps, BQPB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BQPB

!  Arguments

!  ps - given pointer to the structure
!  BQPB_control - BQPB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BQPB_control_type ) :: BQPB_control

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
                                 pc, BQPB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BQPB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BQPB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BQPB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BQPB_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, BQPB_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, BQPB_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, BQPB_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, BQPB_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, BQPB_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, BQPB_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, BQPB_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BQPB_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, BQPB_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, BQPB_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, BQPB_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, BQPB_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, BQPB_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, BQPB_control%stop_rel_c )
        CASE( 'perturb_h' )
          CALL MATLAB_get_value( ps, 'perturb_h',                              &
                                 pc, BQPB_control%perturb_h )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, BQPB_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, BQPB_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, BQPB_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, BQPB_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, BQPB_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, BQPB_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, BQPB_control%reduce_infeas )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, BQPB_control%obj_unbounded )
        CASE( 'potential_unbounded' )
          CALL MATLAB_get_value( ps, 'potential_unbounded',                    &
                                 pc, BQPB_control%potential_unbounded )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, BQPB_control%identical_bounds_tol )
        CASE( 'mu_pounce' )
          CALL MATLAB_get_value( ps, 'mu_pounce',                              &
                                 pc, BQPB_control%mu_pounce )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, BQPB_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, BQPB_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, BQPB_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BQPB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, BQPB_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, BQPB_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, BQPB_control%treat_zero_bounds_as_general )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, BQPB_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, BQPB_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, BQPB_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, BQPB_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, BQPB_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity',          &
                                 pc, BQPB_control%balance_initial_complentarity)
        CASE( 'crossover' )
          CALL MATLAB_get_value( ps, 'crossover',                              &
                                 pc, BQPB_control%crossover )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BQPB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BQPB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BQPB_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, BQPB_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, BQPB_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BQPB_matlab_control_set

      END SUBROUTINE BQPB_matlab_control_set

!-*-   B Q P B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BQPB_matlab_control_get( struct, BQPB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BQPB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQPB_control - BQPB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQPB_control_type ) :: BQPB_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 50
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
         'stop_rel_c                     ', 'perturb_h                      ', &
         'prfeas                         ', 'dufeas                         ', &
         'muzero                         ', 'tau                            ', &
         'gamma_c                        ', 'gamma_f                        ', &
         'reduce_infeas                  ', 'obj_unbounded                  ', &
         'potential_unbounded            ', 'identical_bounds_tol           ', &
         'mu_pounce                      ', 'indicator_tol_p                ', &
         'indicator_tol_pd               ', 'indicator_tol_tapia            ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'remove_dependencies            ',                                    &
         'treat_zero_bounds_as_general   ', 'just_feasible                  ', &
         'getdua                         ', 'puiseux                        ', &
         'every_order                    ', 'feasol                         ', &
         'balance_initial_complentarity  ', 'crossover                      ', &
         'space_critical                 ',                                    &
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
                                  BQPB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BQPB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BQPB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BQPB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BQPB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  BQPB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  BQPB_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  BQPB_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  BQPB_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  BQPB_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  BQPB_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  BQPB_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BQPB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  BQPB_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  BQPB_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  BQPB_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  BQPB_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  BQPB_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  BQPB_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'perturb_h',                        &
                                  BQPB_control%perturb_h )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  BQPB_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  BQPB_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  BQPB_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  BQPB_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  BQPB_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  BQPB_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  BQPB_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  BQPB_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'potential_unbounded',              &
                                  BQPB_control%potential_unbounded )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  BQPB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_pounce',                        &
                                  BQPB_control%mu_pounce )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  BQPB_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  BQPB_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  BQPB_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BQPB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  BQPB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  BQPB_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  BQPB_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  BQPB_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  BQPB_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  BQPB_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  BQPB_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  BQPB_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  BQPB_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'crossover',                        &
                                  BQPB_control%crossover )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BQPB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BQPB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BQPB_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, BQPB_control%FDC_control,          &
                                   'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, BQPB_control%SBLS_control,        &
                                    'SBLS_control' )

      RETURN

!  End of subroutine BQPB_matlab_control_get

      END SUBROUTINE BQPB_matlab_control_get

!-*-  B Q P B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -

      SUBROUTINE BQPB_matlab_inform_create( struct, BQPB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BQPB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQPB_pointer - BQPB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQPB_pointer_type ) :: BQPB_pointer
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
        CALL MATLAB_create_substructure( struct, name, BQPB_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BQPB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'status', BQPB_pointer%status )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
         'alloc_status', BQPB_pointer%alloc_status )
      CALL MATLAB_create_char_component( BQPB_pointer%pointer,                 &
        'bad_alloc', BQPB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'iter', BQPB_pointer%iter )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'factorization_status', BQPB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'factorization_integer', BQPB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'factorization_real', BQPB_pointer%factorization_real )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'nfacts', BQPB_pointer%nfacts )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'nbacts', BQPB_pointer%nbacts )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
        'obj', BQPB_pointer%obj )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
         'primal_infeasibility', BQPB_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
         'dual_infeasibility', BQPB_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
         'complementary_slackness', BQPB_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
        'potential', BQPB_pointer%potential )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
        'non_negligible_pivot', BQPB_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( BQPB_pointer%pointer,              &
        'feasible', BQPB_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BQPB_pointer%pointer,                   &
        'time', BQPB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'total', BQPB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'preprocess', BQPB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'find_dependent', BQPB_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'analyse', BQPB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'factorize', BQPB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'solve', BQPB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_total', BQPB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_preprocess', BQPB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_find_dependent', BQPB_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_analyse', BQPB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_factorize', BQPB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'clock_solve', BQPB_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( BQPB_pointer%pointer,                     &
                                     BQPB_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( BQPB_pointer%pointer,                    &
                                      BQPB_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine BQPB_matlab_inform_create

      END SUBROUTINE BQPB_matlab_inform_create

!-*-*-   B Q P B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE BQPB_matlab_inform_get( BQPB_inform, BQPB_pointer )

!  --------------------------------------------------------------

!  Set BQPB_inform values from matlab pointers

!  Arguments

!  BQPB_inform - BQPB inform structure
!  BQPB_pointer - BQPB pointer structure

!  --------------------------------------------------------------

      TYPE ( BQPB_inform_type ) :: BQPB_inform
      TYPE ( BQPB_pointer_type ) :: BQPB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BQPB_inform%status,                             &
                               mxGetPr( BQPB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%alloc_status,                       &
                               mxGetPr( BQPB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BQPB_pointer%pointer,                           &
                               'bad_alloc', BQPB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BQPB_inform%iter,                               &
                               mxGetPr( BQPB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%factorization_status,               &
                               mxGetPr( BQPB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%factorization_integer,              &
                               mxGetPr( BQPB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%factorization_real,                 &
                               mxGetPr( BQPB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%nfacts,                             &
                               mxGetPr( BQPB_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%nbacts,                             &
                               mxGetPr( BQPB_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%obj,                                &
                               mxGetPr( BQPB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%primal_infeasibility,               &
                               mxGetPr( BQPB_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%dual_infeasibility,                 &
                               mxGetPr( BQPB_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%complementary_slackness,            &
                               mxGetPr( BQPB_pointer%complementary_slackness ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%potential,                          &
                               mxGetPr( BQPB_pointer%potential ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%non_negligible_pivot,               &
                               mxGetPr( BQPB_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%feasible,                           &
                               mxGetPr( BQPB_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%total, wp ),             &
                               mxGetPr( BQPB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%preprocess, wp ),        &
                               mxGetPr( BQPB_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%find_dependent, wp ),    &
                          mxGetPr( BQPB_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%analyse, wp ),           &
                               mxGetPr( BQPB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%factorize, wp ),         &
                               mxGetPr( BQPB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%solve, wp ),             &
                               mxGetPr( BQPB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_total, wp ),       &
                      mxGetPr( BQPB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_preprocess, wp ),  &
                      mxGetPr( BQPB_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_find_dependent,wp),&
                      mxGetPr( BQPB_pointer%time_pointer%clock_find_dependent) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_analyse, wp ),     &
                      mxGetPr( BQPB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_factorize, wp ),   &
                      mxGetPr( BQPB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%clock_solve, wp ),       &
                      mxGetPr( BQPB_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( BQPB_inform%FDC_inform,                      &
                                  BQPB_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SBLS_matlab_inform_get( BQPB_inform%SBLS_inform,                    &
                                   BQPB_pointer%SBLS_pointer )

      RETURN

!  End of subroutine BQPB_matlab_inform_get

      END SUBROUTINE BQPB_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _  B Q P B _ T Y P E S   M O D U L E  -*-*-

    END MODULE GALAHAD_BQPB_MATLAB_TYPES
