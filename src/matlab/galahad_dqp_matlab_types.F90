#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.5 - 20/09/2012 AT 08:00 GMT.

!-*-*-*-  G A L A H A D _ D Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.5. August 1st, 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DQP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to DQP

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_DQP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DQP_matlab_control_set, DQP_matlab_control_get,                &
                DQP_matlab_inform_create, DQP_matlab_inform_get

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

      TYPE, PUBLIC :: DQP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: DQP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts
        mwPointer :: threads, obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness, non_negligible_pivot
        mwPointer :: feasible
        TYPE ( DQP_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
        TYPE ( GLTR_pointer_type ) :: GLTR_pointer
!       TYPE ( ULS_pointer_type ) :: ULS_pointer
      END TYPE
    CONTAINS

!-*-  C Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE DQP_matlab_control_set( ps, DQP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to DQP

!  Arguments

!  ps - given pointer to the structure
!  DQP_control - DQP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( DQP_control_type ) :: DQP_control

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
                                 pc, DQP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, DQP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, DQP_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, DQP_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, DQP_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, DQP_control%print_gap )
        CASE( 'dual_starting_point' )
          CALL MATLAB_get_value( ps, 'dual_starting_point',                    &
                                 pc, DQP_control%dual_starting_point  )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, DQP_control%maxit  )
        CASE( 'max_sc' )
          CALL MATLAB_get_value( ps, 'max_sc',                                 &
                                 pc, DQP_control%max_sc  )
        CASE( 'cauchy_only' )
          CALL MATLAB_get_value( ps, 'cauchy_only',                            &
                                 pc, DQP_control%cauchy_only  )
        CASE( 'arc_search_maxit ' )
          CALL MATLAB_get_value( ps, 'arc_search_maxit ',                      &
                                 pc, DQP_control%arc_search_maxit  )
        CASE( 'cg_maxit' )
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, DQP_control%maxit  )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, DQP_control%restore_problem )
        CASE( 'rho' )
          CALL MATLAB_get_value( ps, 'rho',                                    &
                                 pc, DQP_control%rho )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, DQP_control%infinity )
        CASE( 'stop_abs_p' )
          CALL MATLAB_get_value( ps, 'stop_abs_p',                             &
                                 pc, DQP_control%stop_abs_p )
        CASE( 'stop_rel_p' )
          CALL MATLAB_get_value( ps, 'stop_rel_p',                             &
                                 pc, DQP_control%stop_rel_p )
        CASE( 'stop_abs_d' )
          CALL MATLAB_get_value( ps, 'stop_abs_d',                             &
                                 pc, DQP_control%stop_abs_d )
        CASE( 'stop_rel_d' )
          CALL MATLAB_get_value( ps, 'stop_rel_d',                             &
                                 pc, DQP_control%stop_rel_d )
        CASE( 'stop_abs_c' )
          CALL MATLAB_get_value( ps, 'stop_abs_c',                             &
                                 pc, DQP_control%stop_abs_c )
        CASE( 'stop_rel_c' )
          CALL MATLAB_get_value( ps, 'stop_rel_c',                             &
                                 pc, DQP_control%stop_rel_c )
        CASE( 'stop_cg_relative' )
          CALL MATLAB_get_value( ps, 'stop_cg_relative',                       &
                                 pc, DQP_control%stop_cg_relative )
        CASE( 'stop_cg_absolute' )
          CALL MATLAB_get_value( ps, 'stop_cg_absolute',                       &
                                 pc, DQP_control%stop_cg_absolute )
        CASE( 'cg_zero_curvature' )
          CALL MATLAB_get_value( ps, 'cg_zero_curvature',                      &
                                 pc, DQP_control%cg_zero_curvature )
!       CASE( 'obj_unbounded' )
!         CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
!                                pc, DQP_control%obj_unbounded )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, DQP_control%identical_bounds_tol )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, DQP_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, DQP_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, DQP_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, DQP_control%treat_zero_bounds_as_general )
!       CASE( 'just_feasible' )
!         CALL MATLAB_get_value( ps, 'just_feasible',                          &
!                                pc, DQP_control%just_feasible )
!       CASE( 'getdua' )
!         CALL MATLAB_get_value( ps, 'getdua',                                 &
!                                pc, DQP_control%getdua )
        CASE( 'exact_arc_search' )
          CALL MATLAB_get_value( ps, 'exact_arc_search',                       &
                                 pc, DQP_control%exact_arc_search )
        CASE( 'subspace_direct' )
          CALL MATLAB_get_value( ps, 'subspace_direct',                        &
                                 pc, DQP_control%subspace_direct )
        CASE( 'subspace_arc_search' )
          CALL MATLAB_get_value( ps, 'subspace_arc_search',                    &
                                 pc, DQP_control%subspace_arc_search )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, DQP_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, DQP_control%deallocate_error_fatal )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, DQP_control%symmetric_linear_solver,     &
                                  len )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, DQP_control%definite_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, DQP_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, DQP_control%FDC_control, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, DQP_control%SLS_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, DQP_control%SBLS_control, len )
        CASE( 'GLTR_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLTR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLTR_control must be a structure' )
          CALL GLTR_matlab_control_set( pc, DQP_control%GLTR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine DQP_matlab_control_set

      END SUBROUTINE DQP_matlab_control_set

!-*-  C Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE DQP_matlab_control_get( struct, DQP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to DQP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DQP_control - DQP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DQP_control_type ) :: DQP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 42
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'dual_starting_point            ', 'maxit                          ', &
         'max_sc                         ', 'cauchy_only                    ', &
         'arc_search_maxit               ', 'cg_maxit                       ', &
         'restore_problem                ', 'rho                            ', &
         'infinity                       ', 'stop_abs_p                     ', &
         'stop_rel_p                     ', 'stop_abs_d                     ', &
         'stop_rel_d                     ', 'stop_abs_c                     ', &
         'stop_rel_c                     ', 'stop_cg_relative               ', &
         'stop_cg_absolute               ', 'cg_zero_curvature              ', &
         'identical_bounds_tol           ', 'cpu_time_limit                 ', &
         'clock_time_limit               ', 'remove_dependencies            ', &
         'treat_zero_bounds_as_general   ', 'exact_arc_search               ', &
         'subspace_direct                ', 'subspace_arc_search            ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'symmetric_linear_solver        ', 'definite_linear_solver         ', &
         'unsymmetric_linear_solver      ', 'prefix                         ', &
         'FDC_control                    ', 'SLS_control                    ', &
         'SBLS_control                   ', 'GLTR_control                   ' /)

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
                                  DQP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  DQP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  DQP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  DQP_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  DQP_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  DQP_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'dual_starting_point',              &
                                  DQP_control%dual_starting_point )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  DQP_control%maxit )
      CALL MATLAB_fill_component( pointer, 'max_sc',                           &
                                  DQP_control%max_sc )
      CALL MATLAB_fill_component( pointer, 'cauchy_only',                      &
                                  DQP_control%cauchy_only )
      CALL MATLAB_fill_component( pointer, 'arc_search_maxit',                 &
                                  DQP_control%arc_search_maxit )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  DQP_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  DQP_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'rho',                              &
                                  DQP_control%rho )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  DQP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_abs_p',                       &
                                  DQP_control%stop_abs_p )
      CALL MATLAB_fill_component( pointer, 'stop_rel_p',                       &
                                  DQP_control%stop_rel_p )
      CALL MATLAB_fill_component( pointer, 'stop_abs_d',                       &
                                  DQP_control%stop_abs_d )
      CALL MATLAB_fill_component( pointer, 'stop_rel_d',                       &
                                  DQP_control%stop_rel_d )
      CALL MATLAB_fill_component( pointer, 'stop_abs_c',                       &
                                  DQP_control%stop_abs_c )
      CALL MATLAB_fill_component( pointer, 'stop_rel_c',                       &
                                  DQP_control%stop_rel_c )
      CALL MATLAB_fill_component( pointer, 'stop_cg_relative',                 &
                                  DQP_control%stop_cg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_cg_absolute',                 &
                                  DQP_control%stop_cg_absolute )
      CALL MATLAB_fill_component( pointer, 'cg_zero_curvature',                &
                                  DQP_control%cg_zero_curvature )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  DQP_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  DQP_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  DQP_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  DQP_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  DQP_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'exact_arc_search',                 &
                                  DQP_control%exact_arc_search )
      CALL MATLAB_fill_component( pointer, 'subspace_direct',                  &
                                  DQP_control%subspace_direct )
      CALL MATLAB_fill_component( pointer, 'subspace_arc_search',              &
                                  DQP_control%subspace_arc_search )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  DQP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  DQP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  DQP_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  DQP_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'unsymmetric_linear_solver',        &
                                  DQP_control%unsymmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  DQP_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, DQP_control%FDC_control,           &
                                   'FDC_control' )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, DQP_control%SLS_control,           &
                                   'SLS_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, DQP_control%SBLS_control,         &
                                    'SBLS_control' )

!  create the components of sub-structure GLTR_control

      CALL GLTR_matlab_control_get( pointer, DQP_control%GLTR_control,           &
                                   'GLTR_control' )

      RETURN

!  End of subroutine DQP_matlab_control_get

      END SUBROUTINE DQP_matlab_control_get

!-*- C Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE DQP_matlab_inform_create( struct, DQP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold DQP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  DQP_pointer - DQP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( DQP_pointer_type ) :: DQP_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 20
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'iter                    ',             &
           'factorization_status    ', 'factorization_integer   ',             &
           'factorization_real      ', 'nfacts                  ',             &
           'threads                 ',                                         &
           'obj                     ', 'primal_infeasibility    ',             &
           'dual_infeasibility      ', 'complementary_slackness ',             &
           'non_negligible_pivot    ', 'feasible                ',             &
           'time                    ', 'FDC_inform              ',             &
           'SLS_inform              ', 'SBLS_inform             ',             &
           'GLTR_inform             ' /)
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
        CALL MATLAB_create_substructure( struct, name, DQP_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        DQP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'status', DQP_pointer%status )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
         'alloc_status', DQP_pointer%alloc_status )
      CALL MATLAB_create_char_component( DQP_pointer%pointer,                  &
        'bad_alloc', DQP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'iter', DQP_pointer%iter )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'factorization_status', DQP_pointer%factorization_status )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'factorization_integer', DQP_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'factorization_real', DQP_pointer%factorization_real )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'nfacts', DQP_pointer%nfacts )
      CALL MATLAB_create_integer_component( DQP_pointer%pointer,               &
        'threads', DQP_pointer%threads )
      CALL MATLAB_create_real_component( DQP_pointer%pointer,                  &
        'obj', DQP_pointer%obj )
      CALL MATLAB_create_real_component( DQP_pointer%pointer,                  &
         'primal_infeasibility', DQP_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( DQP_pointer%pointer,                  &
         'dual_infeasibility', DQP_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( DQP_pointer%pointer,                  &
         'complementary_slackness', DQP_pointer%complementary_slackness )
      CALL MATLAB_create_real_component( DQP_pointer%pointer,                  &
        'non_negligible_pivot', DQP_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( DQP_pointer%pointer,               &
        'feasible', DQP_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( DQP_pointer%pointer,                    &
        'time', DQP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'total', DQP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'preprocess', DQP_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'find_dependent', DQP_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'analyse', DQP_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'factorize', DQP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'solve', DQP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_total', DQP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_preprocess', DQP_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_find_dependent', DQP_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_analyse', DQP_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_factorize', DQP_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( DQP_pointer%time_pointer%pointer,     &
        'clock_solve', DQP_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( DQP_pointer%pointer,                      &
                                     DQP_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( DQP_pointer%pointer,                      &
                                     DQP_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( DQP_pointer%pointer,                     &
                                      DQP_pointer%SBLS_pointer, 'SBLS_inform' )

!  Define the components of sub-structure GLTR_inform

      CALL GLTR_matlab_inform_create( DQP_pointer%pointer,                     &
                                      DQP_pointer%GLTR_pointer, 'GLTR_inform' )

      RETURN

!  End of subroutine DQP_matlab_inform_create

      END SUBROUTINE DQP_matlab_inform_create

!-*-*-  C Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE DQP_matlab_inform_get( DQP_inform, DQP_pointer )

!  --------------------------------------------------------------

!  Set DQP_inform values from matlab pointers

!  Arguments

!  DQP_inform - DQP inform structure
!  DQP_pointer - DQP pointer structure

!  --------------------------------------------------------------

      TYPE ( DQP_inform_type ) :: DQP_inform
      TYPE ( DQP_pointer_type ) :: DQP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( DQP_inform%status,                              &
                               mxGetPr( DQP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%alloc_status,                        &
                               mxGetPr( DQP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( DQP_pointer%pointer,                            &
                               'bad_alloc', DQP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( DQP_inform%iter,                                &
                               mxGetPr( DQP_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%factorization_status,                &
                               mxGetPr( DQP_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%factorization_integer,               &
                               mxGetPr( DQP_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%factorization_real,                  &
                               mxGetPr( DQP_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%nfacts,                              &
                               mxGetPr( DQP_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%threads,                             &
                               mxGetPr( DQP_pointer%threads ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%obj,                                 &
                               mxGetPr( DQP_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%primal_infeasibility,                &
                               mxGetPr( DQP_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%dual_infeasibility,                  &
                               mxGetPr( DQP_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%complementary_slackness,             &
                               mxGetPr( DQP_pointer%complementary_slackness ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%non_negligible_pivot,                &
                               mxGetPr( DQP_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( DQP_inform%feasible,                            &
                               mxGetPr( DQP_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%total, wp ),              &
                               mxGetPr( DQP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%preprocess, wp ),         &
                               mxGetPr( DQP_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%find_dependent, wp ),     &
                          mxGetPr( DQP_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%analyse, wp ),            &
                               mxGetPr( DQP_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%factorize, wp ),          &
                               mxGetPr( DQP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%solve, wp ),              &
                               mxGetPr( DQP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_total, wp ),        &
                      mxGetPr( DQP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( DQP_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_find_dependent, wp),&
                      mxGetPr( DQP_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_analyse, wp ),      &
                      mxGetPr( DQP_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_factorize, wp ),    &
                      mxGetPr( DQP_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( DQP_inform%time%clock_solve, wp ),        &
                      mxGetPr( DQP_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( DQP_inform%FDC_inform,                       &
                                  DQP_pointer%FDC_pointer )

!  definite linear solvers

      CALL SLS_matlab_inform_get( DQP_inform%SLS_inform,                       &
                                  DQP_pointer%SLS_pointer )

!  indefinite linear solvers

      CALL SBLS_matlab_inform_get( DQP_inform%SBLS_inform,                     &
                                   DQP_pointer%SBLS_pointer )

!  unsymmetric linear solvers

      CALL GLTR_matlab_inform_get( DQP_inform%GLTR_inform,                     &
                                  DQP_pointer%GLTR_pointer )

      RETURN

!  End of subroutine DQP_matlab_inform_get

      END SUBROUTINE DQP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ C Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_DQP_MATLAB_TYPES
