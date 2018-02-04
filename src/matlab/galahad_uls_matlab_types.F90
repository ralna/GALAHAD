#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 26/02/2010 AT 14:00 GMT.

!-**-*-*-  G A L A H A D _ S L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 10th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_ULS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to ULS

      USE GALAHAD_MATLAB
      USE GALAHAD_ULS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: ULS_matlab_control_set, ULS_matlab_control_get,                &
                ULS_matlab_inform_create, ULS_matlab_inform_get

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

      TYPE, PUBLIC :: ULS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: more_info, out_of_range, duplicates
        mwPointer :: entries_dropped, workspace_factors, compresses
        mwPointer :: entries_in_factors, rank, structural_rank
        mwPointer :: pivot_control, iterative_refinements
      END TYPE 

    CONTAINS

!-*-*-  S L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE ULS_matlab_control_set( ps, ULS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to ULS

!  Arguments

!  ps - given pointer to the structure
!  ULS_control - ULS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( ULS_control_type ) :: ULS_control

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
                                 pc, ULS_control%error )
        CASE( 'warning' )
          CALL MATLAB_get_value( ps, 'warning',                                &
                                 pc, ULS_control%warning )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, ULS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, ULS_control%print_level )
        CASE( 'print_level_solver' )
          CALL MATLAB_get_value( ps, 'print_level_solver',                     &
                                 pc, ULS_control%print_level_solver )
        CASE( 'initial_fill_in_factor' )
          CALL MATLAB_get_value( ps, 'initial_fill_in_factor',                 &
                                 pc, ULS_control%initial_fill_in_factor )
        CASE( 'max_factor_size' )
          CALL galmxGetLong( ps, 'max_factor_size',                            &
                                 pc, ULS_control%max_factor_size )
        CASE( 'blas_block_size_factorize' )
          CALL MATLAB_get_value( ps, 'blas_block_size_factorize',              &
                                 pc, ULS_control%blas_block_size_factorize )
        CASE( 'blas_block_size_solve' )
          CALL MATLAB_get_value( ps, 'blas_block_size_solve',                  &
                                 pc, ULS_control%blas_block_size_solve )
        CASE( 'pivot_control' )
          CALL MATLAB_get_value( ps, 'pivot_control',                          &
                                 pc, ULS_control%pivot_control )
        CASE( 'pivot_search_limit' )
          CALL MATLAB_get_value( ps, 'pivot_search_limit',                     &
                                 pc, ULS_control%pivot_search_limit )
        CASE( 'minimum_size_for_btf' )
          CALL MATLAB_get_value( ps, 'minimum_size_for_btf',                   &
                                 pc, ULS_control%minimum_size_for_btf )
        CASE( 'max_iterative_refinements' )
          CALL MATLAB_get_value( ps, 'max_iterative_refinements',              &
                                 pc, ULS_control%max_iterative_refinements )
        CASE( 'stop_if_singular' )
          CALL MATLAB_get_value( ps, 'stop_if_singular',                       &
                                 pc, ULS_control%stop_if_singular )
        CASE( 'array_increase_factor' )
          CALL MATLAB_get_value( ps, 'array_increase_factor',                  &
                                 pc, ULS_control%array_increase_factor )
        CASE( 'array_decrease_factor' )
          CALL MATLAB_get_value( ps, 'array_decrease_factor',                  &
                                 pc, ULS_control%array_decrease_factor )
        CASE( 'switch_to_full_code_density' )
          CALL MATLAB_get_value( ps, 'switch_to_full_code_density',            &
                                 pc, ULS_control%switch_to_full_code_density )
        CASE( 'relative_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'relative_pivot_tolerance',               &
                                 pc, ULS_control%relative_pivot_tolerance )
        CASE( 'absolute_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'absolute_pivot_tolerance',               &
                                 pc, ULS_control%absolute_pivot_tolerance )
        CASE( 'zero_tolerance' )
          CALL MATLAB_get_value( ps, 'zero_tolerance',                         &
                                 pc, ULS_control%zero_tolerance )
        CASE( 'acceptable_residual_relative' )
          CALL MATLAB_get_value( ps,'acceptable_residual_relative',            &
                                 pc, ULS_control%acceptable_residual_relative )
        CASE( 'acceptable_residual_absolute' )
          CALL MATLAB_get_value( ps,'acceptable_residual_absolute',            &
                                 pc, ULS_control%acceptable_residual_absolute )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, ULS_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine ULS_matlab_control_set

      END SUBROUTINE ULS_matlab_control_set

!-*-*-  U L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE ULS_matlab_control_get( struct, ULS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to ULS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  ULS_control - ULS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ULS_control_type ) :: ULS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 23
      CHARACTER ( LEN = 28 ), PARAMETER :: finform( ninform ) = (/             &
           'error                       ', 'warning                     ',     &
           'out                         ', 'print_level                 ',     &
           'print_level_solver          ', 'initial_fill_in_factor      ',     &
           'max_factor_size             ', 'blas_block_size_factorize   ',     &
           'blas_block_size_solve       ', 'pivot_control               ',     &
           'pivot_search_limit          ', 'minimum_size_for_btf        ',     &
           'max_iterative_refinements   ', 'stop_if_singular            ',     &
           'array_increase_factor       ', 'array_decrease_factor       ',     &
           'switch_to_full_code_density ', 'relative_pivot_tolerance    ',     &
           'absolute_pivot_tolerance    ', 'zero_tolerance              ',     &
           'acceptable_residual_relative', 'acceptable_residual_absolute',     &
           'prefix                      '                       /)

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
                                  ULS_control%error )
      CALL MATLAB_fill_component( pointer, 'warning',                          &
                                  ULS_control%warning )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  ULS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  ULS_control%print_level )
      CALL MATLAB_fill_component( pointer,'print_level_solver',                &
                                  ULS_control%print_level_solver )
      CALL MATLAB_fill_component( pointer, 'initial_fill_in_factor',           &
                                  ULS_control%initial_fill_in_factor )
      CALL MATLAB_fill_long_component( pointer, 'max_factor_size',             &
                                  ULS_control%max_factor_size )
      CALL MATLAB_fill_component( pointer, 'blas_block_size_factorize',        &
                                  ULS_control%blas_block_size_factorize )
      CALL MATLAB_fill_component( pointer, 'blas_block_size_solve',            &
                                  ULS_control%blas_block_size_solve )
      CALL MATLAB_fill_component( pointer, 'pivot_control',                    &
                                  ULS_control%pivot_control )
      CALL MATLAB_fill_component( pointer, 'pivot_search_limit',               &
                                  ULS_control%pivot_search_limit )
      CALL MATLAB_fill_component( pointer, 'minimum_size_for_btf',             &
                                  ULS_control%minimum_size_for_btf )
      CALL MATLAB_fill_component( pointer, 'max_iterative_refinements',        &
                                  ULS_control%max_iterative_refinements )
      CALL MATLAB_fill_component( pointer, 'stop_if_singular',                 &
                                  ULS_control%stop_if_singular )
      CALL MATLAB_fill_component( pointer, 'array_increase_factor',            &
                                  ULS_control%array_increase_factor )
      CALL MATLAB_fill_component( pointer, 'array_decrease_factor',            &
                                  ULS_control%array_decrease_factor )
      CALL MATLAB_fill_component( pointer, 'switch_to_full_code_density',      &
                                  ULS_control%switch_to_full_code_density )
      CALL MATLAB_fill_component( pointer, 'relative_pivot_tolerance',         &
                                  ULS_control%relative_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'absolute_pivot_tolerance',         &
                                  ULS_control%absolute_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'zero_tolerance',                   &
                                  ULS_control%zero_tolerance )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_relative',     &
                                  ULS_control%acceptable_residual_relative )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_absolute',     &
                                  ULS_control%acceptable_residual_absolute )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  ULS_control%prefix )

      RETURN

!  End of subroutine ULS_matlab_control_get

      END SUBROUTINE ULS_matlab_control_get

!-*-  S L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -*-

      SUBROUTINE ULS_matlab_inform_create( struct, ULS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold ULS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  name - name of component of the structure
!  ULS_pointer - ULS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ULS_pointer_type ) :: ULS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 14
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'more_info               ',             &
           'out_of_range            ', 'duplicates              ',             &
           'entries_dropped         ', 'workspace_factors       ',             &
           'compresses              ', 'entries_in_factors      ',             &
           'rank                    ', 'structural_rank         ',             &
           'pivot_control           ', 'iterative_refinements   ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, ULS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        ULS_pointer%pointer = struct
      END IF

!  create the components

      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'status', ULS_pointer%status )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'alloc_status', ULS_pointer%alloc_status )
      CALL MATLAB_create_char_component( ULS_pointer%pointer,                  &
        'bad_alloc', ULS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'more_info', ULS_pointer%more_info )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'out_of_range', ULS_pointer%out_of_range )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'duplicates', ULS_pointer%duplicates )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'entries_dropped', ULS_pointer%entries_dropped )
      CALL MATLAB_create_long_component( ULS_pointer%pointer,                  &
        'workspace_factors ', ULS_pointer%workspace_factors  )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'compresses', ULS_pointer%compresses )
      CALL MATLAB_create_long_component( ULS_pointer%pointer,                  &
        'entries_in_factors', ULS_pointer%entries_in_factors )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'rank', ULS_pointer%rank )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'structural_rank', ULS_pointer%structural_rank )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'pivot_control', ULS_pointer%pivot_control )
      CALL MATLAB_create_integer_component( ULS_pointer%pointer,               &
        'iterative_refinements', ULS_pointer%iterative_refinements )

      RETURN

!  End of subroutine ULS_matlab_inform_create

      END SUBROUTINE ULS_matlab_inform_create

!-*-*-  S L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE ULS_matlab_inform_get( ULS_inform, ULS_pointer )

!  --------------------------------------------------------------

!  Set ULS_inform values from matlab pointers

!  Arguments

!  ULS_inform - ULS inform structure
!  ULS_pointer - ULS pointer structure

!  --------------------------------------------------------------

      TYPE ( ULS_inform_type ) :: ULS_inform
      TYPE ( ULS_pointer_type ) :: ULS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( ULS_inform%status,                              &
            mxGetPr( ULS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%alloc_status,                        &
            mxGetPr( ULS_pointer%alloc_status ) )
      CALL  MATLAB_copy_to_ptr( ULS_pointer%pointer,                           &
            'bad_alloc', ULS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( ULS_inform%more_info,                           &
            mxGetPr( ULS_pointer%more_info ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%out_of_range,                        &
            mxGetPr( ULS_pointer%out_of_range ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%duplicates,                          &
            mxGetPr( ULS_pointer%duplicates ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%entries_dropped,                     &
            mxGetPr( ULS_pointer%entries_dropped ) )
      CALL galmxCopyLongToPtr( ULS_inform% workspace_factors,                  &
            mxGetPr( ULS_pointer% workspace_factors  ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%compresses,                          &
            mxGetPr( ULS_pointer%compresses ) )
      CALL galmxCopyLongToPtr( ULS_inform%entries_in_factors,                  &
            mxGetPr( ULS_pointer%entries_in_factors ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%rank,                                &
            mxGetPr( ULS_pointer%rank ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%structural_rank,                     &
            mxGetPr( ULS_pointer%structural_rank ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%pivot_control,                       &
            mxGetPr( ULS_pointer%pivot_control ) )
      CALL MATLAB_copy_to_ptr( ULS_inform%iterative_refinements,               &
            mxGetPr( ULS_pointer%iterative_refinements ) )

      RETURN

!  End of subroutine ULS_matlab_inform_get

      END SUBROUTINE ULS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_ULS_MATLAB_TYPES
