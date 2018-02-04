#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 26/02/2010 AT 14:00 GMT.

!-**-*-*-  G A L A H A D _ S L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 10th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SLS

      USE GALAHAD_MATLAB
      USE GALAHAD_SLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SLS_matlab_control_set, SLS_matlab_control_get,                &
                SLS_matlab_inform_create, SLS_matlab_inform_get

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

      TYPE, PUBLIC :: SLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: entries, out_of_range, duplicates, upper
        mwPointer :: missing_diagonals, more_info
        mwPointer :: max_depth_assembly_tree, nodes_assembly_tree
        mwPointer :: real_size_desirable, integer_size_desirable
        mwPointer :: real_size_necessary, integer_size_necessary
        mwPointer :: real_size_factors, integer_size_factors
        mwPointer :: entries_in_factors, max_task_pool_size
        mwPointer :: max_front_size, compresses_real
        mwPointer :: compresses_integer, two_by_two_pivots, semi_bandwidth
        mwPointer :: delayed_pivots, pivot_sign_changes
        mwPointer :: static_pivots, first_modified_pivot, rank
        mwPointer :: negative_eigenvalues, iterative_refinements
        mwPointer :: flops_assembly, flops_elimination
        mwPointer :: flops_blas, largest_modified_pivot
        mwPointer :: minimum_scaling_factor, maximum_scaling_factor
        mwPointer :: condition_number_1, condition_number_2
        mwPointer :: backward_error_1, backward_error_2, forward_error
      END TYPE 

    CONTAINS

!-*-*-  S L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE SLS_matlab_control_set( ps, SLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SLS

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SLS_control_type ) :: SLS_control

!  local variables

      INTEGER :: i, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO i = 1, nfields
        name = mxGetFieldNameByNumber( ps, i )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, SLS_control%error )
        CASE( 'warning' )
          CALL MATLAB_get_value( ps, 'warning',                                &
                                 pc, SLS_control%warning )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SLS_control%out )
        CASE( 'statistics' )
          CALL MATLAB_get_value( ps, 'statistics',                             &
                                 pc, SLS_control%statistics )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SLS_control%print_level )
        CASE( 'print_level_solver' )
          CALL MATLAB_get_value( ps, 'print_level_solver',                     &
                                 pc, SLS_control%print_level_solver )
        CASE( 'block_size_kernel' )
          CALL MATLAB_get_value( ps, 'block_size_kernel',                      &
                                 pc, SLS_control%block_size_kernel )
        CASE( 'bits' )
          CALL MATLAB_get_value( ps, 'bits',                                   &
                                 pc, SLS_control%bits )
        CASE( 'block_size_elimination' )
          CALL MATLAB_get_value( ps, 'block_size_elimination',                 &
                                 pc, SLS_control%block_size_elimination )
        CASE( 'blas_block_size_factorize' )
          CALL MATLAB_get_value( ps, 'blas_block_size_factorize',              &
                                 pc, SLS_control%blas_block_size_factorize )
        CASE( 'blas_block_size_solve' )
          CALL MATLAB_get_value( ps, 'blas_block_size_solve',                  &
                                 pc, SLS_control%blas_block_size_solve )
        CASE( 'node_amalgamation' )
          CALL MATLAB_get_value( ps, 'node_amalgamation',                      &
                                 pc, SLS_control%node_amalgamation )
        CASE( 'initial_pool_size' )
          CALL MATLAB_get_value( ps, 'initial_pool_size',                      &
                                 pc, SLS_control%initial_pool_size )
        CASE( 'min_real_factor_size' )
          CALL MATLAB_get_value( ps, 'min_real_factor_size',                   &
                                 pc, SLS_control%min_real_factor_size )
        CASE( 'min_integer_factor_size' )
          CALL MATLAB_get_value( ps, 'min_integer_factor_size',                &
                                 pc, SLS_control%min_integer_factor_size )
        CASE( 'max_real_factor_size' )
          CALL galmxGetLong( ps, 'max_real_factor_size',                       &
                                 pc, SLS_control%max_real_factor_size )
        CASE( 'max_integer_factor_size' )
          CALL galmxGetLong( ps, 'max_integer_factor_size',                    &
                                 pc, SLS_control%max_integer_factor_size )
        CASE( 'max_in_core_store' )
          CALL galmxGetLong( ps, 'max_in_core_store',                          &
                                 pc, SLS_control%max_in_core_store )
        CASE( 'pivot_control' )
          CALL MATLAB_get_value( ps, 'pivot_control',                          &
                                 pc, SLS_control%pivot_control )
        CASE( 'ordering' )
          CALL MATLAB_get_value( ps, 'ordering',                               &
                                 pc, SLS_control%ordering )
        CASE( 'full_row_threshold' )
          CALL MATLAB_get_value( ps, 'full_row_threshold',                     &
                                 pc, SLS_control%full_row_threshold )
        CASE( 'row_search_indefinite' )
          CALL MATLAB_get_value( ps, 'row_search_indefinite',                  &
                                 pc, SLS_control%row_search_indefinite )
        CASE( 'scaling' )
          CALL MATLAB_get_value( ps, 'scaling',                                &
                                 pc, SLS_control%scaling )
        CASE( 'scale_maxit' )
          CALL MATLAB_get_value( ps, 'scale_maxit',                            &
                                 pc, SLS_control%scale_maxit )
        CASE( 'scale_thesh' )
          CALL MATLAB_get_value( ps, 'scale_thresh',                           &
                                 pc, SLS_control%scale_thresh )
        CASE( 'max_iterative_refinements' )
          CALL MATLAB_get_value( ps, 'max_iterative_refinements',              &
                                 pc, SLS_control%max_iterative_refinements )
        CASE( 'array_increase_factor' )
          CALL MATLAB_get_value( ps, 'array_increase_factor',                  &
                                 pc, SLS_control%array_increase_factor )
        CASE( 'array_decrease_factor' )
          CALL MATLAB_get_value( ps, 'array_decrease_factor',                  &
                                 pc, SLS_control%array_decrease_factor )
        CASE( 'relative_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'relative_pivot_tolerance',               &
                                 pc, SLS_control%relative_pivot_tolerance )
        CASE( 'minimum_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'minimum_pivot_tolerance',                &
                                 pc, SLS_control%minimum_pivot_tolerance )
        CASE( 'absolute_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'absolute_pivot_tolerance',               &
                                 pc, SLS_control%absolute_pivot_tolerance )
        CASE( 'zero_tolerance' )
          CALL MATLAB_get_value( ps, 'zero_tolerance',                         &
                                 pc, SLS_control%zero_tolerance )
        CASE( 'static_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'static_pivot_tolerance',                 &
                                 pc, SLS_control%static_pivot_tolerance )
        CASE( 'static_level_switch' )
          CALL MATLAB_get_value( ps, 'static_level_switch',                    &
                                 pc, SLS_control%static_level_switch )
        CASE( 'consistency_tolerance' )
          CALL MATLAB_get_value( ps, 'consistency_tolerance',                  &
                                 pc, SLS_control%consistency_tolerance )
        CASE( 'acceptable_residual_relative' )
          CALL MATLAB_get_value( ps,'acceptable_residual_relative',            &
                                 pc, SLS_control%acceptable_residual_relative )
        CASE( 'acceptable_residual_absolute' )
          CALL MATLAB_get_value( ps,'acceptable_residual_absolute',            &
                                 pc, SLS_control%acceptable_residual_absolute )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SLS_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SLS_matlab_control_set

      END SUBROUTINE SLS_matlab_control_set

!-*-*-  S L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE SLS_matlab_control_get( struct, SLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SLS_control - SLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLS_control_type ) :: SLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 38
      CHARACTER ( LEN = 28 ), PARAMETER :: finform( ninform ) = (/             &
           'error                       ', 'warning                     ',     &
           'out                         ', 'statistics                  ',     &
           'print_level                 ', 'print_level_solver          ',     &
           'block_size_kernel           ', 'bits                        ',     &
           'block_size_elimination      ', 'blas_block_size_factorize   ',     &
           'blas_block_size_solve       ', 'node_amalgamation           ',     &
           'initial_pool_size           ', 'min_real_factor_size        ',     &
           'min_integer_factor_size     ', 'max_real_factor_size        ',     &
           'max_integer_factor_size     ', 'max_in_core_store           ',     &
           'pivot_control               ', 'ordering                    ',     &
           'full_row_threshold          ', 'row_search_indefinite       ',     &
           'scaling                     ', 'scale_maxit                 ',     &
           'scale_thresh                ', 'max_iterative_refinements   ',     &
           'array_increase_factor       ', 'array_decrease_factor       ',     &
           'relative_pivot_tolerance    ', 'minimum_pivot_tolerance     ',     &
           'absolute_pivot_tolerance    ', 'zero_tolerance              ',     &
           'static_pivot_tolerance      ', 'static_level_switch         ',     &
           'consistency_tolerance       ', 'acceptable_residual_relative',     &
           'acceptable_residual_absolute', 'prefix                      '     /)

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
                                  SLS_control%error )
      CALL MATLAB_fill_component( pointer, 'warning',                          &
                                  SLS_control%warning )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SLS_control%out )
      CALL MATLAB_fill_component( pointer, 'statistics',                       &
                                  SLS_control%statistics )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SLS_control%print_level )
      CALL MATLAB_fill_component( pointer,'print_level_solver',                &
                                  SLS_control%print_level_solver )
      CALL MATLAB_fill_component( pointer, 'block_size_kernel',                &
                                  SLS_control%block_size_kernel )
      CALL MATLAB_fill_component( pointer, 'bits',                             &
                                  SLS_control%bits )
      CALL MATLAB_fill_component( pointer, 'block_size_elimination',           &
                                  SLS_control%block_size_elimination )
      CALL MATLAB_fill_component( pointer, 'blas_block_size_factorize',        &
                                  SLS_control%blas_block_size_factorize )
      CALL MATLAB_fill_component( pointer, 'blas_block_size_solve',            &
                                  SLS_control%blas_block_size_solve )
      CALL MATLAB_fill_component( pointer, 'node_amalgamation',                &
                                  SLS_control%node_amalgamation )
      CALL MATLAB_fill_component( pointer, 'initial_pool_size',                &
                                  SLS_control%initial_pool_size )
      CALL MATLAB_fill_component( pointer, 'min_real_factor_size',             &
                                  SLS_control%min_real_factor_size )
      CALL MATLAB_fill_component( pointer, 'min_integer_factor_size',          &
                                  SLS_control%min_integer_factor_size )
      CALL MATLAB_fill_long_component( pointer, 'max_real_factor_size',        &
                                  SLS_control%max_real_factor_size )
      CALL MATLAB_fill_long_component( pointer, 'max_integer_factor_size',     &
                                  SLS_control%max_integer_factor_size )
      CALL MATLAB_fill_long_component( pointer, 'max_in_core_store',           &
                                  SLS_control%max_in_core_store )
      CALL MATLAB_fill_component( pointer, 'pivot_control',                    &
                                  SLS_control%pivot_control )
      CALL MATLAB_fill_component( pointer, 'ordering',                         &
                                  SLS_control%ordering )
      CALL MATLAB_fill_component( pointer, 'full_row_threshold',               &
                                  SLS_control%full_row_threshold )
      CALL MATLAB_fill_component( pointer, 'row_search_indefinite',            &
                                  SLS_control%row_search_indefinite )
      CALL MATLAB_fill_component( pointer, 'scaling',                          &
                                  SLS_control%scaling )
      CALL MATLAB_fill_component( pointer, 'scale_maxit',                      &
                                  SLS_control%scale_maxit )
      CALL MATLAB_fill_component( pointer, 'scale_thresh',                     &
                                  SLS_control%scale_thresh )
      CALL MATLAB_fill_component( pointer, 'max_iterative_refinements',        &
                                  SLS_control%max_iterative_refinements )
      CALL MATLAB_fill_component( pointer, 'array_increase_factor',            &
                                  SLS_control%array_increase_factor )
      CALL MATLAB_fill_component( pointer, 'array_decrease_factor',            &
                                  SLS_control%array_decrease_factor )
      CALL MATLAB_fill_component( pointer, 'relative_pivot_tolerance',         &
                                  SLS_control%relative_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'minimum_pivot_tolerance',          &
                                  SLS_control%minimum_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'absolute_pivot_tolerance',         &
                                  SLS_control%absolute_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'zero_tolerance',                   &
                                  SLS_control%zero_tolerance )
      CALL MATLAB_fill_component( pointer, 'static_pivot_tolerance',           &
                                  SLS_control%static_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'static_level_switch',              &
                                  SLS_control%static_level_switch )
      CALL MATLAB_fill_component( pointer, 'consistency_tolerance',            &
                                  SLS_control%consistency_tolerance )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_relative',     &
                                  SLS_control%acceptable_residual_relative )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_absolute',     &
                                  SLS_control%acceptable_residual_absolute )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SLS_control%prefix )

      RETURN

!  End of subroutine SLS_matlab_control_get

      END SUBROUTINE SLS_matlab_control_get

!-*-  S L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE SLS_matlab_inform_create( struct, SLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  name - name of component of the structure
!  SLS_pointer - SLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLS_pointer_type ) :: SLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 42
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'more_info               ',             &
           'entries                 ',                                         &
           'out_of_range            ', 'duplicates              ',             &
           'upper                   ', 'missing_diagonals       ',             &
           'max_depth_assembly_tree ', 'nodes_assembly_tree     ',             &
           'real_size_desirable     ', 'integer_size_desirable  ',             &
           'real_size_necessary     ', 'integer_size_necessary  ',             &
           'real_size_factors       ', 'integer_size_factors    ',             &
           'entries_in_factors      ', 'max_task_pool_size      ',             &
           'max_front_size          ', 'compresses_real         ',             &
           'compresses_integer      ', 'two_by_two_pivots       ',             &
           'semi_bandwidth          ',                                         &
           'delayed_pivots          ', 'pivot_sign_changes      ',             &
           'static_pivots           ', 'first_modified_pivot    ',             &
           'rank                    ', 'negative_eigenvalues    ',             &
           'iterative_refinements   ', 'flops_assembly          ',             &
           'flops_elimination       ', 'flops_blas              ',             &
           'largest_modified_pivot  ', 'minimum_scaling_factor  ',             &
           'maximum_scaling_factor  ', 'condition_number_1      ',             &
           'condition_number_2      ', 'backward_error_1        ',             &
           'backward_error_2        ', 'forward_error           ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SLS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SLS_pointer%pointer = struct
      END IF

!  create the components

      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'status', SLS_pointer%status )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'alloc_status', SLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( SLS_pointer%pointer,                  &
        'bad_alloc', SLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'more_info', SLS_pointer%more_info )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'entries', SLS_pointer%entries )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'out_of_range', SLS_pointer%out_of_range )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'duplicates', SLS_pointer%duplicates )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'upper', SLS_pointer%upper )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'missing_diagonals', SLS_pointer%missing_diagonals )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'max_depth_assembly_tree', SLS_pointer%max_depth_assembly_tree )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'nodes_assembly_tree', SLS_pointer%nodes_assembly_tree )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'real_size_desirable', SLS_pointer%real_size_desirable )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'integer_size_desirable', SLS_pointer%integer_size_desirable )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'real_size_necessary', SLS_pointer%real_size_necessary )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'integer_size_necessary', SLS_pointer%integer_size_necessary )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'real_size_factors ', SLS_pointer%real_size_factors )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'integer_size_factors', SLS_pointer%integer_size_factors )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'entries_in_factors', SLS_pointer%entries_in_factors )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'max_task_pool_size', SLS_pointer%max_task_pool_size )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'max_front_size', SLS_pointer%max_front_size )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'compresses_real', SLS_pointer%compresses_real )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'compresses_integer', SLS_pointer%compresses_integer )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'two_by_two_pivots', SLS_pointer%two_by_two_pivots )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'semi_bandwidth', SLS_pointer%semi_bandwidth )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'delayed_pivots', SLS_pointer%delayed_pivots )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'pivot_sign_changes', SLS_pointer%pivot_sign_changes )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'static_pivots', SLS_pointer%static_pivots )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'first_modified_pivot', SLS_pointer%first_modified_pivot )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'rank', SLS_pointer%rank )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'negative_eigenvalues', SLS_pointer%negative_eigenvalues )
      CALL MATLAB_create_integer_component( SLS_pointer%pointer,               &
        'iterative_refinements', SLS_pointer%iterative_refinements )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'flops_assembly', SLS_pointer%flops_assembly )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'flops_elimination', SLS_pointer%flops_elimination )
      CALL MATLAB_create_long_component( SLS_pointer%pointer,                  &
        'flops_blas', SLS_pointer%flops_blas )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'largest_modified_pivot', SLS_pointer%largest_modified_pivot )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'minimum_scaling_factor', SLS_pointer%minimum_scaling_factor )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'maximum_scaling_factor', SLS_pointer%maximum_scaling_factor )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'condition_number_1', SLS_pointer%condition_number_1 )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'condition_number_2', SLS_pointer%condition_number_2 )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'backward_error_1', SLS_pointer%backward_error_1 )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'backward_error_2', SLS_pointer%backward_error_2 )
      CALL MATLAB_create_real_component( SLS_pointer%pointer,                  &
        'forward_error', SLS_pointer%forward_error )
      RETURN

!  End of subroutine SLS_matlab_inform_create

      END SUBROUTINE SLS_matlab_inform_create

!-*-*-  S L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE SLS_matlab_inform_get( SLS_inform, SLS_pointer )

!  --------------------------------------------------------------

!  Set SLS_inform values from matlab pointers

!  Arguments

!  SLS_inform - SLS inform structure
!  SLS_pointer - SLS pointer structure

!  --------------------------------------------------------------

      TYPE ( SLS_inform_type ) :: SLS_inform
      TYPE ( SLS_pointer_type ) :: SLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SLS_inform%status,                              &
            mxGetPr( SLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%alloc_status,                        &
            mxGetPr( SLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SLS_pointer%pointer,                            &
            'bad_alloc', SLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SLS_inform%more_info,                           &
            mxGetPr( SLS_pointer%more_info ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%entries,                             &
            mxGetPr( SLS_pointer%entries ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%out_of_range,                        &
            mxGetPr( SLS_pointer%out_of_range ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%duplicates,                          &
            mxGetPr( SLS_pointer%duplicates ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%upper,                               &
            mxGetPr( SLS_pointer%upper ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%missing_diagonals,                   &
            mxGetPr( SLS_pointer%missing_diagonals ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%max_depth_assembly_tree,             &
            mxGetPr( SLS_pointer%max_depth_assembly_tree ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%nodes_assembly_tree,                 &
            mxGetPr( SLS_pointer%nodes_assembly_tree ) )
      CALL galmxCopyLongToPtr( SLS_inform%real_size_desirable,                 &
            mxGetPr( SLS_pointer%real_size_desirable ) )
      CALL galmxCopyLongToPtr( SLS_inform%integer_size_desirable,              &
            mxGetPr( SLS_pointer%integer_size_desirable ) )
      CALL galmxCopyLongToPtr( SLS_inform%real_size_necessary,                 &
            mxGetPr( SLS_pointer%real_size_necessary ) )
      CALL galmxCopyLongToPtr( SLS_inform%integer_size_necessary,              &
            mxGetPr( SLS_pointer%integer_size_necessary ) )
      CALL galmxCopyLongToPtr( SLS_inform%real_size_factors,                   &
            mxGetPr( SLS_pointer%real_size_factors ) )
      CALL galmxCopyLongToPtr( SLS_inform%integer_size_factors,                &
            mxGetPr( SLS_pointer%integer_size_factors ) )
      CALL galmxCopyLongToPtr( SLS_inform%entries_in_factors,                  &
            mxGetPr( SLS_pointer%entries_in_factors ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%max_task_pool_size,                  &
            mxGetPr( SLS_pointer%max_task_pool_size ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%max_front_size,                      &
            mxGetPr( SLS_pointer%max_front_size ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%compresses_real,                     &
            mxGetPr( SLS_pointer%compresses_real ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%compresses_integer,                  &
            mxGetPr( SLS_pointer%compresses_integer ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%two_by_two_pivots,                   &
            mxGetPr( SLS_pointer%two_by_two_pivots ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%semi_bandwidth,                      &
            mxGetPr( SLS_pointer%semi_bandwidth ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%delayed_pivots,                      &
            mxGetPr( SLS_pointer%delayed_pivots ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%pivot_sign_changes,                  &
            mxGetPr( SLS_pointer%pivot_sign_changes ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%static_pivots,                       &
            mxGetPr( SLS_pointer%static_pivots ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%first_modified_pivot,                &
            mxGetPr( SLS_pointer%first_modified_pivot ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%rank,                                &
            mxGetPr( SLS_pointer%rank ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%negative_eigenvalues,                &
            mxGetPr( SLS_pointer%negative_eigenvalues ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%iterative_refinements,               &
            mxGetPr( SLS_pointer%iterative_refinements ) )
      CALL galmxCopyLongToPtr( SLS_inform%flops_assembly,                      &
            mxGetPr( SLS_pointer%flops_assembly ) )
      CALL galmxCopyLongToPtr( SLS_inform%flops_elimination,                   &
            mxGetPr( SLS_pointer%flops_elimination ) )
      CALL galmxCopyLongToPtr( SLS_inform%flops_blas,                          &
            mxGetPr( SLS_pointer%flops_blas ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%largest_modified_pivot,              &
            mxGetPr( SLS_pointer%largest_modified_pivot ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%minimum_scaling_factor,              &
            mxGetPr( SLS_pointer%minimum_scaling_factor ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%maximum_scaling_factor,              &
            mxGetPr( SLS_pointer%maximum_scaling_factor ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%condition_number_1,                  &
            mxGetPr( SLS_pointer%condition_number_1 ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%condition_number_2,                  &
            mxGetPr( SLS_pointer%condition_number_2 ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%backward_error_1,                    &
            mxGetPr( SLS_pointer%backward_error_1 ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%backward_error_2,                    &
            mxGetPr( SLS_pointer%backward_error_2 ) )
      CALL MATLAB_copy_to_ptr( SLS_inform%forward_error,                       &
            mxGetPr( SLS_pointer%forward_error ) )

      RETURN

!  End of subroutine SLS_matlab_inform_get

      END SUBROUTINE SLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SLS_MATLAB_TYPES

