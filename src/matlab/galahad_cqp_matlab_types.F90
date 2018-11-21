#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 04/03/2011 AT 18:00 GMT.

!-*-*-*-  G A L A H A D _ C Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CQP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to CQP

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_CQP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CQP_matlab_control_set, CQP_matlab_control_get,                &
                CQP_matlab_inform_create, CQP_matlab_inform_get

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

      TYPE, PUBLIC :: CQP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: CQP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts, nbacts
        mwPointer :: obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, potential, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( CQP_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE
    CONTAINS

!-*-  C Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE CQP_matlab_control_set( ps, CQP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to CQP

!  Arguments

!  ps - given pointer to the structure
!  CQP_control - CQP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( CQP_control_type ) :: CQP_control

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
                                 pc, CQP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, CQP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, CQP_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, CQP_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, CQP_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, CQP_control%maxit  )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, CQP_control%infeas_max  )
        CASE( 'muzero_fixed' )
          CALL MATLAB_get_value( ps, 'muzero_fixed',                           &
                                 pc, CQP_control%muzero_fixed )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, CQP_control%restore_problem )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, CQP_control%indicator_type )
        CASE( 'arc' )
          CALL MATLAB_get_value( ps, 'arc',                                    &
                                 pc, CQP_control%arc )
        CASE( 'series_order' )
          CALL MATLAB_get_value( ps, 'series_order',                           &
                                 pc, CQP_control%series_order )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, CQP_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, CQP_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, CQP_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, CQP_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, CQP_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, CQP_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, CQP_control%stop_rel_c )
        CASE( 'perturb_h' )
          CALL MATLAB_get_value( ps, 'perturb_h',                              &
                                 pc, CQP_control%perturb_h )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, CQP_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, CQP_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, CQP_control%muzero )
        CASE( 'tau' )
          CALL MATLAB_get_value( ps, 'tau',                                    &
                                 pc, CQP_control%tau )
        CASE( 'gamma_c' )
          CALL MATLAB_get_value( ps, 'gamma_c',                                &
                                 pc, CQP_control%gamma_c )
        CASE( 'gamma_f' )
          CALL MATLAB_get_value( ps, 'gamma_f',                                &
                                 pc, CQP_control%gamma_f )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, CQP_control%reduce_infeas )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, CQP_control%obj_unbounded )
        CASE( 'potential_unbounded' )
          CALL MATLAB_get_value( ps, 'potential_unbounded',                    &
                                 pc, CQP_control%potential_unbounded )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, CQP_control%identical_bounds_tol )
        CASE( 'mu_lunge' )
          CALL MATLAB_get_value( ps, 'mu_lunge',                               &
                                 pc, CQP_control%mu_lunge )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, CQP_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, CQP_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, CQP_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, CQP_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, CQP_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, CQP_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, CQP_control%treat_zero_bounds_as_general )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, CQP_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, CQP_control%getdua )
        CASE( 'puiseux' )
          CALL MATLAB_get_value( ps, 'puiseux',                                &
                                 pc, CQP_control%puiseux )
        CASE( 'every_order' )
          CALL MATLAB_get_value( ps, 'every_order',                            &
                                 pc, CQP_control%every_order )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, CQP_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity',          &
                                 pc, CQP_control%balance_initial_complentarity )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, CQP_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, CQP_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, CQP_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, CQP_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, CQP_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine CQP_matlab_control_set

      END SUBROUTINE CQP_matlab_control_set

!-*-  C Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE CQP_matlab_control_get( struct, CQP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to CQP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  CQP_control - CQP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( CQP_control_type ) :: CQP_control
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
         'stop_rel_c                     ', 'perturb_h                      ', &
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
                                  CQP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  CQP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  CQP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  CQP_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  CQP_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  CQP_control%maxit )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  CQP_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'muzero_fixed',                     &
                                  CQP_control%muzero_fixed )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  CQP_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  CQP_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'arc',                              &
                                  CQP_control%arc )
      CALL MATLAB_fill_component( pointer, 'series_order',                     &
                                  CQP_control%series_order )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  CQP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  CQP_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  CQP_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  CQP_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  CQP_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  CQP_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  CQP_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'perturb_h',                        &
                                  CQP_control%perturb_h )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  CQP_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  CQP_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  CQP_control%muzero )
      CALL MATLAB_fill_component( pointer, 'tau',                              &
                                  CQP_control%tau )
      CALL MATLAB_fill_component( pointer, 'gamma_c',                          &
                                  CQP_control%gamma_c )
      CALL MATLAB_fill_component( pointer, 'gamma_f',                          &
                                  CQP_control%gamma_f )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  CQP_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  CQP_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'potential_unbounded',              &
                                  CQP_control%potential_unbounded )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  CQP_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_lunge',                         &
                                  CQP_control%mu_lunge )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  CQP_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  CQP_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  CQP_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  CQP_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  CQP_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  CQP_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  CQP_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  CQP_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  CQP_control%getdua )
      CALL MATLAB_fill_component( pointer, 'puiseux',                          &
                                  CQP_control%puiseux )
      CALL MATLAB_fill_component( pointer, 'every_order',                      &
                                  CQP_control%every_order )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  CQP_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  CQP_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  CQP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  CQP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  CQP_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, CQP_control%FDC_control,           &
                                   'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, CQP_control%SBLS_control,         &
                                    'SBLS_control' )

      RETURN

!  End of subroutine CQP_matlab_control_get

      END SUBROUTINE CQP_matlab_control_get

!-*- C Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE CQP_matlab_inform_create( struct, CQP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold CQP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  CQP_pointer - CQP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( CQP_pointer_type ) :: CQP_pointer
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
        CALL MATLAB_create_substructure( struct, name, CQP_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        CQP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'status', CQP_pointer%status )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
         'alloc_status', CQP_pointer%alloc_status )
      CALL MATLAB_create_char_component( CQP_pointer%pointer,                  &
        'bad_alloc', CQP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'iter', CQP_pointer%iter )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'factorization_status', CQP_pointer%factorization_status )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'factorization_integer', CQP_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'factorization_real', CQP_pointer%factorization_real )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'nfacts', CQP_pointer%nfacts )
      CALL MATLAB_create_integer_component( CQP_pointer%pointer,               &
        'nbacts', CQP_pointer%nbacts )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
        'obj', CQP_pointer%obj )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
         'primal_infeasibility', CQP_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
         'dual_infeasibility', CQP_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
         'complementary_slackness', CQP_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
        'potential', CQP_pointer%potential )
      CALL MATLAB_create_real_component( CQP_pointer%pointer,                  &
        'non_negligible_pivot', CQP_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( CQP_pointer%pointer,               &
        'feasible', CQP_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( CQP_pointer%pointer,                    &
        'time', CQP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'total', CQP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'preprocess', CQP_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'find_dependent', CQP_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'analyse', CQP_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'factorize', CQP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'solve', CQP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_total', CQP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_preprocess', CQP_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_find_dependent', CQP_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_analyse', CQP_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_factorize', CQP_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( CQP_pointer%time_pointer%pointer,     &
        'clock_solve', CQP_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( CQP_pointer%pointer,                      &
                                     CQP_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( CQP_pointer%pointer,                     &
                                      CQP_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine CQP_matlab_inform_create

      END SUBROUTINE CQP_matlab_inform_create

!-*-*-  C Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE CQP_matlab_inform_get( CQP_inform, CQP_pointer )

!  --------------------------------------------------------------

!  Set CQP_inform values from matlab pointers

!  Arguments

!  CQP_inform - CQP inform structure
!  CQP_pointer - CQP pointer structure

!  --------------------------------------------------------------

      TYPE ( CQP_inform_type ) :: CQP_inform
      TYPE ( CQP_pointer_type ) :: CQP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( CQP_inform%status,                              &
                               mxGetPr( CQP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%alloc_status,                        &
                               mxGetPr( CQP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( CQP_pointer%pointer,                            &
                               'bad_alloc', CQP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( CQP_inform%iter,                                &
                               mxGetPr( CQP_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%factorization_status,                &
                               mxGetPr( CQP_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%factorization_integer,               &
                               mxGetPr( CQP_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%factorization_real,                  &
                               mxGetPr( CQP_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%nfacts,                              &
                               mxGetPr( CQP_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%nbacts,                              &
                               mxGetPr( CQP_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%obj,                                 &
                               mxGetPr( CQP_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%primal_infeasibility,                &
                               mxGetPr( CQP_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%dual_infeasibility,                  &
                               mxGetPr( CQP_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%complementary_slackness,             &
                               mxGetPr( CQP_pointer%complementary_slackness ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%potential,                           &
                               mxGetPr( CQP_pointer%potential ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%non_negligible_pivot,                &
                               mxGetPr( CQP_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( CQP_inform%feasible,                            &
                               mxGetPr( CQP_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%total, wp ),              &
                               mxGetPr( CQP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%preprocess, wp ),         &
                               mxGetPr( CQP_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%find_dependent, wp ),     &
                          mxGetPr( CQP_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%analyse, wp ),            &
                               mxGetPr( CQP_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%factorize, wp ),          &
                               mxGetPr( CQP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%solve, wp ),              &
                               mxGetPr( CQP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_total, wp ),        &
                      mxGetPr( CQP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( CQP_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_find_dependent, wp),&
                      mxGetPr( CQP_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_analyse, wp ),      &
                      mxGetPr( CQP_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_factorize, wp ),    &
                      mxGetPr( CQP_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( CQP_inform%time%clock_solve, wp ),        &
                      mxGetPr( CQP_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( CQP_inform%FDC_inform,                       &
                                  CQP_pointer%FDC_pointer )

!  indefinite linear solvers

      CALL SBLS_matlab_inform_get( CQP_inform%SBLS_inform,                     &
                                   CQP_pointer%SBLS_pointer )

      RETURN

!  End of subroutine CQP_matlab_inform_get

      END SUBROUTINE CQP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ C Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_CQP_MATLAB_TYPES
