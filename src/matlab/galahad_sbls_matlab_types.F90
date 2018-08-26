#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 01/02/2011 AT 18:30 GMT.

!-*-*-*-  G A L A H A D _ S B L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 12th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SBLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SBLS

      USE GALAHAD_MATLAB
      USE GALAHAD_ULS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_SBLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SBLS_matlab_control_set, SBLS_matlab_control_get,              &
                SBLS_matlab_inform_create, SBLS_matlab_inform_get

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

      TYPE, PUBLIC :: SBLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, form, factorize, apply
        mwPointer :: clock_total, clock_form, clock_factorize, clock_apply
      END TYPE

      TYPE, PUBLIC :: SBLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: sort_status, factorization_integer, factorization_real
        mwPointer :: preconditioner, factorization, d_plus, rank, rank_def
        mwPointer :: perturbed, iter_pcg, norm_residual
        TYPE ( SBLS_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( ULS_pointer_type ) :: ULS_pointer
      END TYPE
    CONTAINS

!-*-  S B L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE SBLS_matlab_control_set( ps, SBLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SBLS

!  Arguments

!  ps - given pointer to the structure
!  SBLS_control - SBLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SBLS_control_type ) :: SBLS_control

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
                                 pc, SBLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SBLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SBLS_control%print_level )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, SBLS_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, SBLS_control%valmin )
        CASE( 'len_ulsmin' )
          CALL MATLAB_get_value( ps, 'len_ulsmin',                             &
                                 pc, SBLS_control%len_ulsmin )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, SBLS_control%itref_max )
        CASE( 'maxit_pcg' )
          CALL MATLAB_get_value( ps, 'maxit_pcg',                              &
                                 pc, SBLS_control%maxit_pcg )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, SBLS_control%new_a )
        CASE( 'new_h' )
          CALL MATLAB_get_value( ps, 'new_h',                                  &
                                 pc, SBLS_control%new_h )
        CASE( 'new_c' )
          CALL MATLAB_get_value( ps, 'new_c',                                  &
                                 pc, SBLS_control%new_c )
        CASE( 'preconditioner' )
          CALL MATLAB_get_value( ps, 'preconditioner',                         &
                                 pc, SBLS_control%preconditioner )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, SBLS_control%semi_bandwidth )
        CASE( 'factorization' )
          CALL MATLAB_get_value( ps, 'factorization',                          &
                                 pc, SBLS_control%factorization )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, SBLS_control%max_col )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, SBLS_control%pivot_tol )
        CASE( 'pivot_tol_for_basis' )
          CALL MATLAB_get_value( ps, 'pivot_tol_for_basis',                    &
                                 pc, SBLS_control%pivot_tol_for_basis )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, SBLS_control%zero_pivot )
        CASE( 'min_diagonal' )
          CALL MATLAB_get_value( ps, 'min_diagonal',                           &
                                 pc, SBLS_control%min_diagonal )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, SBLS_control%stop_absolute )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, SBLS_control%stop_relative )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, SBLS_control%remove_dependencies )
        CASE( 'find_basis_by_transpose' )
          CALL MATLAB_get_value( ps, 'find_basis_by_transpose',                &
                                 pc, SBLS_control%find_basis_by_transpose )
        CASE( 'affine' )
          CALL MATLAB_get_value( ps, 'affine',                                 &
                                 pc, SBLS_control%affine )
        CASE( 'perturb_to_make_definite' )
          CALL MATLAB_get_value( ps, 'perturb_to_make_definite',               &
                                 pc, SBLS_control%perturb_to_make_definite )
        CASE( 'get_norm_residual' )
          CALL MATLAB_get_value( ps, 'get_norm_residual',                      &
                                 pc, SBLS_control%get_norm_residual )
        CASE( 'check_basis' )
          CALL MATLAB_get_value( ps, 'check_basis',                            &
                                 pc, SBLS_control%check_basis )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SBLS_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, SBLS_control%deallocate_error_fatal  )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, SBLS_control%symmetric_linear_solver,    &
                                  len )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, SBLS_control%definite_linear_solver, len )
        CASE( 'unsymmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'unsymmetric_linear_solver',             &
                                  pc, SBLS_control%unsymmetric_linear_solver,  &
                                  len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SBLS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, SBLS_control%SLS_control, len )
        CASE( 'ULS_control' )
          pc = mxGetField( ps, 1_mwi_, 'ULS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component ULS_control must be a structure' )
          CALL ULS_matlab_control_set( pc, SBLS_control%ULS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SBLS_matlab_control_set

      END SUBROUTINE SBLS_matlab_control_set

!-*-  S B L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE SBLS_matlab_control_get( struct, SBLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SBLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SBLS_control - SBLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SBLS_control_type ) :: SBLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 35
      CHARACTER ( LEN = 25 ), PARAMETER :: finform( ninform ) = (/             &
           'error                    ', 'out                      ',           &
           'print_level              ', 'indmin                   ',           &
           'valmin                   ', 'len_ulsmin               ',           &
           'itref_max                ', 'maxit_pcg                ',           &
           'new_a                    ',                                        &
           'new_h                    ', 'new_c                    ',           &
           'preconditioner           ', 'semi_bandwidth           ',           &
           'factorization            ', 'max_col                  ',           &
           'pivot_tol                ', 'pivot_tol_for_basis      ',           &
           'zero_pivot               ', 'min_diagonal             ',           &
           'stop_absolute            ', 'stop_relative            ',           &
           'remove_dependencies      ', 'find_basis_by_transpose  ',           &
           'affine                   ', 'perturb_to_make_definite ',           &
           'get_norm_residual        ', 'check_basis              ',           &
           'space_critical           ', 'deallocate_error_fatal   ',           &
           'symmetric_linear_solver  ', 'definite_linear_solver   ',           &
           'unsymmetric_linear_solver', 'prefix                   ',           &
           'SLS_control              ', 'ULS_control              ' /)

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
                                  SBLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SBLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SBLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  SBLS_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  SBLS_control%valmin )
      CALL MATLAB_fill_component( pointer, 'len_ulsmin',                       &
                                  SBLS_control%len_ulsmin )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  SBLS_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'maxit_pcg',                        &
                                  SBLS_control%maxit_pcg )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  SBLS_control%new_a )
      CALL MATLAB_fill_component( pointer, 'new_h',                            &
                                  SBLS_control%new_h )
      CALL MATLAB_fill_component( pointer, 'new_c',                            &
                                  SBLS_control%new_c )
      CALL MATLAB_fill_component( pointer, 'preconditioner',                   &
                                  SBLS_control%preconditioner )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  SBLS_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'factorization',                    &
                                  SBLS_control%factorization )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  SBLS_control%max_col )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  SBLS_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_basis',              &
                                  SBLS_control%pivot_tol_for_basis )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  SBLS_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'min_diagonal',                     &
                                  SBLS_control%min_diagonal )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  SBLS_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  SBLS_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  SBLS_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'find_basis_by_transpose',          &
                                  SBLS_control%find_basis_by_transpose )
      CALL MATLAB_fill_component( pointer, 'affine',                           &
                                  SBLS_control%affine )
      CALL MATLAB_fill_component( pointer, 'perturb_to_make_definite',         &
                                  SBLS_control%perturb_to_make_definite )
      CALL MATLAB_fill_component( pointer, 'get_norm_residual',                &
                                  SBLS_control%get_norm_residual )
      CALL MATLAB_fill_component( pointer, 'check_basis',                      &
                                  SBLS_control%check_basis )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SBLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  SBLS_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  SBLS_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  SBLS_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'unsymmetric_linear_solver',        &
                                  SBLS_control%unsymmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SBLS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, SBLS_control%SLS_control,          &
                                  'SLS_control' )

!  create the components of sub-structure ULS_control

      CALL ULS_matlab_control_get( pointer, SBLS_control%ULS_control,          &
                                  'ULS_control' )

      RETURN

!  End of subroutine SBLS_matlab_control_get

      END SUBROUTINE SBLS_matlab_control_get

!-*- S B L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE SBLS_matlab_inform_create( struct, SBLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SBLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SBLS_pointer - SBLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 17
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'sort_status          ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'preconditioner       ', 'factorization        ',                   &
           'd_plus               ', 'rank                 ',                   &
           'rank_def             ', 'perturbed            ',                   &
           'iter_pcg             ',                                            &
           'norm_residual        ', 'time                 ',                   &
           'SLS_inform           ', 'ULS_inform           ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'form                 ',                   &
           'factorize            ', 'apply                ',                   &
           'clock_total          ', 'clock_form           ',                   &
           'clock_factorize      ', 'clock_apply          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SBLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SBLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
        'status', SBLS_pointer%status )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'alloc_status', SBLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( SBLS_pointer%pointer,                 &
        'bad_alloc', SBLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'sort_status', SBLS_pointer%sort_status )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'factorization_integer', SBLS_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'factorization_real', SBLS_pointer%factorization_real )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'preconditioner', SBLS_pointer%preconditioner )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'factorization', SBLS_pointer%factorization )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'd_plus', SBLS_pointer%d_plus )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'rank', SBLS_pointer%rank )
      CALL MATLAB_create_logical_component( SBLS_pointer%pointer,              &
         'rank_def', SBLS_pointer%rank_def )
      CALL MATLAB_create_logical_component( SBLS_pointer%pointer,              &
         'perturbed', SBLS_pointer%perturbed )
      CALL MATLAB_create_integer_component( SBLS_pointer%pointer,              &
         'iter_pcg', SBLS_pointer%iter_pcg )
      CALL MATLAB_create_real_component( SBLS_pointer%pointer,                 &
         'norm_residual', SBLS_pointer%norm_residual )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( SBLS_pointer%pointer,                   &
        'time', SBLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'total', SBLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'form', SBLS_pointer%time_pointer%form )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'factorize', SBLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'apply', SBLS_pointer%time_pointer%apply )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'clock_total', SBLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'clock_form', SBLS_pointer%time_pointer%clock_form )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'clock_factorize', SBLS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( SBLS_pointer%time_pointer%pointer,    &
        'clock_apply', SBLS_pointer%time_pointer%clock_apply )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( SBLS_pointer%pointer,                     &
                                     SBLS_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure ULS_inform

      CALL ULS_matlab_inform_create( SBLS_pointer%pointer,                     &
                                     SBLS_pointer%ULS_pointer, 'ULS_inform' )

      RETURN

!  End of subroutine SBLS_matlab_inform_create

      END SUBROUTINE SBLS_matlab_inform_create

!-*-*  S B L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   *-*-

      SUBROUTINE SBLS_matlab_inform_get( SBLS_inform, SBLS_pointer )

!  --------------------------------------------------------------

!  Set SBLS_inform values from matlab pointers

!  Arguments

!  SBLS_inform - SBLS inform structure
!  SBLS_pointer - SBLS pointer structure

!  --------------------------------------------------------------

      TYPE ( SBLS_inform_type ) :: SBLS_inform
      TYPE ( SBLS_pointer_type ) :: SBLS_pointer

!     INTEGER ::  mexPrintf
!     integer*4 out
!     CHARACTER ( LEN = 200 ) :: str

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SBLS_inform%status,                             &
                               mxGetPr( SBLS_pointer%status ) )
! WRITE( str, "( ' alloc_status'  )" )
! out = mexPrintf( TRIM( str ) // achar(10) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%alloc_status,                       &
                               mxGetPr( SBLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SBLS_pointer%pointer,                           &
                               'bad_alloc', SBLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SBLS_inform%sort_status,                        &
                               mxGetPr( SBLS_pointer%sort_status ) )
      CALL galmxCopyLongToPtr( SBLS_inform%factorization_integer,              &
                               mxGetPr( SBLS_pointer%factorization_integer ) )
      CALL galmxCopyLongToPtr( SBLS_inform%factorization_real,                 &
                               mxGetPr( SBLS_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%preconditioner,                     &
                               mxGetPr( SBLS_pointer%preconditioner ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%factorization,                      &
                               mxGetPr( SBLS_pointer%factorization ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%d_plus,                             &
                               mxGetPr( SBLS_pointer%d_plus ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%rank,                               &
                               mxGetPr( SBLS_pointer%rank ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%rank_def,                           &
                               mxGetPr( SBLS_pointer%rank_def ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%perturbed,                          &
                               mxGetPr( SBLS_pointer%perturbed ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%iter_pcg,                           &
                               mxGetPr( SBLS_pointer%iter_pcg ) )
      CALL MATLAB_copy_to_ptr( SBLS_inform%norm_residual,                      &
                               mxGetPr( SBLS_pointer%norm_residual ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%total, wp ),             &
                               mxGetPr( SBLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%form, wp ),              &
                               mxGetPr( SBLS_pointer%time_pointer%form ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%factorize, wp ),         &
                               mxGetPr( SBLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%apply, wp ),             &
                               mxGetPr( SBLS_pointer%time_pointer%apply ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%clock_total, wp ),       &
                          mxGetPr( SBLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%clock_form, wp ),        &
                          mxGetPr( SBLS_pointer%time_pointer%clock_form ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%clock_factorize, wp ),   &
                          mxGetPr( SBLS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SBLS_inform%time%clock_apply, wp ),       &
                          mxGetPr( SBLS_pointer%time_pointer%clock_apply ) )

!  symmetric linear system components

      CALL SLS_matlab_inform_get( SBLS_inform%SLS_inform,                      &
                                  SBLS_pointer%SLS_pointer )

!  unsymmetric linear system components

      CALL ULS_matlab_inform_get( SBLS_inform%ULS_inform,                      &
                                  SBLS_pointer%ULS_pointer )

      RETURN

!  End of subroutine SBLS_matlab_inform_get

      END SUBROUTINE SBLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S B L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SBLS_MATLAB_TYPES
