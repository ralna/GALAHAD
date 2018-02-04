#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 15:15 GMT.

!-*-*-*-  G A L A H A D _ S P L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_PSLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to PSLS

      USE GALAHAD_MATLAB
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_PSLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: PSLS_matlab_control_set, PSLS_matlab_control_get,              &
                PSLS_matlab_inform_create, PSLS_matlab_inform_get

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

      TYPE, PUBLIC :: PSLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve, update
      END TYPE 

      TYPE, PUBLIC :: PSLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: analyse_status, factorize_status, solve_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: preconditioner, semi_bandwidth, semi_bandwidth_used
        mwPointer :: neg1, neg2, perturbed, fill_in_ratio, norm_residual
        TYPE ( PSLS_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
      END TYPE 
    CONTAINS

!-*-  S P L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE PSLS_matlab_control_set( ps, PSLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to PSLS

!  Arguments

!  ps - given pointer to the structure
!  PSLS_control - PSLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( PSLS_control_type ) :: PSLS_control

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
                                 pc, PSLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, PSLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, PSLS_control%print_level )
        CASE( 'preconditioner' )
          CALL MATLAB_get_value( ps, 'preconditioner',                         &
                                 pc, PSLS_control%preconditioner )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, PSLS_control%semi_bandwidth )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, PSLS_control%max_col )
        CASE( 'icfs_vectors' )
          CALL MATLAB_get_value( ps, 'icfs_vectors',                           &
                                 pc, PSLS_control%icfs_vectors )
        CASE( 'min_diagonal' )
          CALL MATLAB_get_value( ps, 'min_diagonal',                           &
                                 pc, PSLS_control%min_diagonal )
        CASE( 'new_structure' )
          CALL MATLAB_get_value( ps, 'new_structure',                          &
                                 pc, PSLS_control%new_structure )
        CASE( 'get_semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'get_semi_bandwidth',                     &
                                 pc, PSLS_control%get_semi_bandwidth )
        CASE( 'get_norm_residual' )
          CALL MATLAB_get_value( ps, 'get_norm_residual',                      &
                                 pc, PSLS_control%get_norm_residual )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, PSLS_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, PSLS_control%deallocate_error_fatal  )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, PSLS_control%definite_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, PSLS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, PSLS_control%SLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine PSLS_matlab_control_set

      END SUBROUTINE PSLS_matlab_control_set

!-*-  S P L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE PSLS_matlab_control_get( struct, PSLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to PSLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  PSLS_control - PSLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( PSLS_control_type ) :: PSLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'preconditioner                 ', &
         'semi_bandwidth                 ', 'max_col                        ', &
         'icfs_vectors                   ',                                    &
         'min_diagonal                   ', 'new_structure                  ', &
         'get_semi_bandwidth             ', 'get_norm_residual              ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'definite_linear_solver         ', 'prefix                         ', &
         'SLS_control                    '                      /)

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
                                  PSLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  PSLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  PSLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'preconditioner',                   &
                                  PSLS_control%preconditioner )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  PSLS_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  PSLS_control%max_col )
      CALL MATLAB_fill_component( pointer, 'icfs_vectors',                     &
                                  PSLS_control%icfs_vectors )
      CALL MATLAB_fill_component( pointer, 'min_diagonal',                     &
                                  PSLS_control%min_diagonal )
      CALL MATLAB_fill_component( pointer, 'new_structure',                    &
                                  PSLS_control%new_structure )
      CALL MATLAB_fill_component( pointer, 'get_semi_bandwidth',               &
                                  PSLS_control%get_semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'get_norm_residual',                &
                                  PSLS_control%get_norm_residual )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  PSLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  PSLS_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  PSLS_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  PSLS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, PSLS_control%SLS_control,          &
                                   'SLS_control' )

      RETURN

!  End of subroutine PSLS_matlab_control_get

      END SUBROUTINE PSLS_matlab_control_get

!-*- S P L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE PSLS_matlab_inform_create( struct, PSLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold PSLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  PSLS_pointer - PSLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( PSLS_pointer_type ) :: PSLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 18
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'analyse_status       ',                   &
           'factorize_status     ', 'solve_status         ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'preconditioner       ', 'semi_bandwidth       ',                   &
           'semi_bandwidth_used  ', 'neg1                 ',                   &
           'neg2                 ', 'perturbed            ',                   &
           'fill_in_ratio        ', 'norm_residual        ',                   &
           'time                 ', 'SLS_inform           '     /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'assemble             ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'update               '          /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, PSLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        PSLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
        'status', PSLS_pointer%status )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'alloc_status', PSLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( PSLS_pointer%pointer,                 &
        'bad_alloc', PSLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'analyse_status', PSLS_pointer%analyse_status )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'factorize_status', PSLS_pointer%factorize_status )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'solve_status', PSLS_pointer%solve_status )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'factorization_integer', PSLS_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'factorization_real', PSLS_pointer%factorization_real )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'preconditioner', PSLS_pointer%preconditioner )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'semi_bandwidth', PSLS_pointer%semi_bandwidth )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'semi_bandwidth_used', PSLS_pointer%semi_bandwidth_used )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'neg1', PSLS_pointer%neg1 )
      CALL MATLAB_create_integer_component( PSLS_pointer%pointer,              &
         'neg2', PSLS_pointer%neg2 )
      CALL MATLAB_create_logical_component( PSLS_pointer%pointer,              &
         'perturbed', PSLS_pointer%perturbed )
      CALL MATLAB_create_real_component( PSLS_pointer%pointer,                 &
         'fill_in_ratio', PSLS_pointer%fill_in_ratio )
      CALL MATLAB_create_real_component( PSLS_pointer%pointer,                 &
         'norm_residual', PSLS_pointer%norm_residual )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( PSLS_pointer%pointer,                   &
        'time', PSLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'total', PSLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'assemble', PSLS_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'analyse', PSLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'factorize', PSLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'solve', PSLS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( PSLS_pointer%time_pointer%pointer,    &
        'update', PSLS_pointer%time_pointer%update )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( PSLS_pointer%pointer,                     &
                                     PSLS_pointer%SLS_pointer, 'SLS_inform' )

      RETURN

!  End of subroutine PSLS_matlab_inform_create

      END SUBROUTINE PSLS_matlab_inform_create

!-*-*-  S P L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE PSLS_matlab_inform_get( PSLS_inform, PSLS_pointer )

!  --------------------------------------------------------------

!  Set PSLS_inform values from matlab pointers

!  Arguments

!  PSLS_inform - PSLS inform structure
!  PSLS_pointer - PSLS pointer structure

!  --------------------------------------------------------------

      TYPE ( PSLS_inform_type ) :: PSLS_inform
      TYPE ( PSLS_pointer_type ) :: PSLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( PSLS_inform%status,                             &
                               mxGetPr( PSLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%alloc_status,                       &
                               mxGetPr( PSLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( PSLS_pointer%pointer,                           &
                               'bad_alloc', PSLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( PSLS_inform%analyse_status,                     &
                               mxGetPr( PSLS_pointer%analyse_status ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%factorize_status,                   &
                               mxGetPr( PSLS_pointer%factorize_status ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%solve_status,                       &
                               mxGetPr( PSLS_pointer%solve_status ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%factorization_integer,              &
                               mxGetPr( PSLS_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%factorization_real,                 &
                               mxGetPr( PSLS_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%preconditioner,                     &
                               mxGetPr( PSLS_pointer%preconditioner ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%semi_bandwidth,                     &
                               mxGetPr( PSLS_pointer%semi_bandwidth ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%semi_bandwidth_used,                &
                               mxGetPr( PSLS_pointer%semi_bandwidth_used ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%neg1,                               &
                               mxGetPr( PSLS_pointer%neg1 ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%neg2,                               &
                               mxGetPr( PSLS_pointer%neg2 ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%perturbed,                          &
                               mxGetPr( PSLS_pointer%perturbed ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%fill_in_ratio,                      &
                               mxGetPr( PSLS_pointer%fill_in_ratio ) )
      CALL MATLAB_copy_to_ptr( PSLS_inform%norm_residual,                      &
                               mxGetPr( PSLS_pointer%norm_residual ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%total, wp ),             &
                               mxGetPr( PSLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%assemble, wp ),          &
                               mxGetPr( PSLS_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%analyse, wp ),           &
                               mxGetPr( PSLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%factorize, wp ),         &
                               mxGetPr( PSLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%solve, wp ),             &
                               mxGetPr( PSLS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( PSLS_inform%time%update, wp ),             &
                               mxGetPr( PSLS_pointer%time_pointer%update ) )

!  symmetric linear system components

      CALL SLS_matlab_inform_get( PSLS_inform%SLS_inform,                      &
                                  PSLS_pointer%SLS_pointer )

      RETURN

!  End of subroutine PSLS_matlab_inform_get

      END SUBROUTINE PSLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S P L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_PSLS_MATLAB_TYPES
