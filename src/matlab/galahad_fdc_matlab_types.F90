#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 04/03/2011 AT 16:00 GMT.

!-*-*-*-  G A L A H A D _ F D C _ M A T L A B _ T Y P E S   M O D U L E  -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. March 4th, 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_FDC_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to FDC

      USE GALAHAD_MATLAB
      USE GALAHAD_ULS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_FDC_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: FDC_matlab_control_set, FDC_matlab_control_get,                &
                FDC_matlab_inform_create, FDC_matlab_inform_get

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

      TYPE, PUBLIC :: FDC_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize
        mwPointer :: clock_total, clock_analyse, clock_factorize
      END TYPE 

      TYPE, PUBLIC :: FDC_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: factorization_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: non_negligible_pivot
        TYPE ( FDC_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( ULS_pointer_type ) :: ULS_pointer
      END TYPE 
    CONTAINS

!-*-  F D C _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE FDC_matlab_control_set( ps, FDC_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to FDC

!  Arguments

!  ps - given pointer to the structure
!  FDC_control - FDC control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( FDC_control_type ) :: FDC_control

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
                                 pc, FDC_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, FDC_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, FDC_control%print_level )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, FDC_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, FDC_control%valmin )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, FDC_control%pivot_tol )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, FDC_control%zero_pivot )
        CASE( 'max_infeas' )
          CALL MATLAB_get_value( ps, 'max_infeas',                             &
                                 pc, FDC_control%max_infeas )
        CASE( 'use_sls' )
          CALL MATLAB_get_value( ps, 'use_sls',                                &
                                 pc, FDC_control%use_sls )
        CASE( 'scale' )
          CALL MATLAB_get_value( ps, 'scale',                                  &
                                 pc, FDC_control%scale )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, FDC_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, FDC_control%deallocate_error_fatal  )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, FDC_control%symmetric_linear_solver,     &
                                  len )
        CASE( 'unsymmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'unsymmetric_linear_solver',             &
                                  pc, FDC_control%unsymmetric_linear_solver,   &
                                  len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, FDC_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, FDC_control%SLS_control, len )
        CASE( 'ULS_control' )
          pc = mxGetField( ps, 1_mwi_, 'ULS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component ULS_control must be a structure' )
          CALL ULS_matlab_control_set( pc, FDC_control%ULS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine FDC_matlab_control_set

      END SUBROUTINE FDC_matlab_control_set

!-*-  F D C _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE FDC_matlab_control_get( struct, FDC_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to FDC

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  FDC_control - FDC control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( FDC_control_type ) :: FDC_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 18
      CHARACTER ( LEN = 25 ), PARAMETER :: finform( ninform ) = (/             &
           'error                    ', 'out                      ',           &
           'print_level              ', 'indmin                   ',           &
           'valmin                   ', 'len_ulsmin               ',           &
           'pivot_tol                ', 'zero_pivot               ',           &
           'max_infeas               ', 'use_sls                  ',           &
           'scale                    ', 'space_critical           ',           &
           'deallocate_error_fatal   ', 'symmetric_linear_solver  ',           &
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
                                  FDC_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  FDC_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  FDC_control%print_level )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  FDC_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  FDC_control%valmin )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  FDC_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  FDC_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'max_infeas',                       &
                                  FDC_control%max_infeas )
      CALL MATLAB_fill_component( pointer, 'use_sls',                          &
                                  FDC_control%use_sls )
      CALL MATLAB_fill_component( pointer, 'scale',                            &
                                  FDC_control%scale )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  FDC_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  FDC_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  FDC_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'unsymmetric_linear_solver',        &
                                  FDC_control%unsymmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  FDC_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, FDC_control%SLS_control,           &
                                  'SLS_control' )

!  create the components of sub-structure ULS_control

      CALL ULS_matlab_control_get( pointer, FDC_control%ULS_control,           &
                                  'ULS_control' )

      RETURN

!  End of subroutine FDC_matlab_control_get

      END SUBROUTINE FDC_matlab_control_get

!-*- F D C _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE FDC_matlab_inform_create( struct, FDC_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold FDC_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  FDC_pointer - FDC pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( FDC_pointer_type ) :: FDC_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_status ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'non_negligible_pivot ', 'time                 ',                   &
           'SLS_inform           ', 'ULS_inform           ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'analyse              ',                   &
           'factorize            ', 'clock_total          ',                   &
           'clock_analyse        ', 'clock_factorize      ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, FDC_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        FDC_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( FDC_pointer%pointer,               &
        'status', FDC_pointer%status )
      CALL MATLAB_create_integer_component( FDC_pointer%pointer,               &
         'alloc_status', FDC_pointer%alloc_status )
      CALL MATLAB_create_char_component( FDC_pointer%pointer,                  &
        'bad_alloc', FDC_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( FDC_pointer%pointer,               &
         'factorization_status', FDC_pointer%factorization_status )
      CALL MATLAB_create_integer_component( FDC_pointer%pointer,               &
         'factorization_integer', FDC_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( FDC_pointer%pointer,               &
         'factorization_real', FDC_pointer%factorization_real )
      CALL MATLAB_create_real_component( FDC_pointer%pointer,                  &
         'non_negligible_pivot', FDC_pointer%non_negligible_pivot )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( FDC_pointer%pointer,                    &
        'time', FDC_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'total', FDC_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'analyse', FDC_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'factorize', FDC_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'clock_total', FDC_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'clock_analyse', FDC_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( FDC_pointer%time_pointer%pointer,     &
        'clock_factorize', FDC_pointer%time_pointer%clock_factorize )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( FDC_pointer%pointer,                      &
                                     FDC_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure ULS_inform

      CALL ULS_matlab_inform_create( FDC_pointer%pointer,                      &
                                     FDC_pointer%ULS_pointer, 'ULS_inform' )

      RETURN

!  End of subroutine FDC_matlab_inform_create

      END SUBROUTINE FDC_matlab_inform_create

!-*-*  F D C _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   *-*-

      SUBROUTINE FDC_matlab_inform_get( FDC_inform, FDC_pointer )

!  --------------------------------------------------------------

!  Set FDC_inform values from matlab pointers

!  Arguments

!  FDC_inform - FDC inform structure
!  FDC_pointer - FDC pointer structure

!  --------------------------------------------------------------

      TYPE ( FDC_inform_type ) :: FDC_inform
      TYPE ( FDC_pointer_type ) :: FDC_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( FDC_inform%status,                              &
                               mxGetPr( FDC_pointer%status ) )
      CALL MATLAB_copy_to_ptr( FDC_inform%alloc_status,                        &
                               mxGetPr( FDC_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( FDC_pointer%pointer,                            &
                               'bad_alloc', FDC_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( FDC_inform%factorization_status,                &
                               mxGetPr( FDC_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( FDC_inform%factorization_integer,               &
                               mxGetPr( FDC_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( FDC_inform%factorization_real,                  &
                               mxGetPr( FDC_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( FDC_inform%non_negligible_pivot,                &
                               mxGetPr( FDC_pointer%non_negligible_pivot ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%total, wp ),              &
                               mxGetPr( FDC_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%analyse, wp ),            &
                               mxGetPr( FDC_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%factorize, wp ),          &
                               mxGetPr( FDC_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%clock_total, wp ),        &
                          mxGetPr( FDC_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%clock_analyse, wp ),      &
                          mxGetPr( FDC_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( FDC_inform%time%clock_factorize, wp ),    &
                          mxGetPr( FDC_pointer%time_pointer%clock_factorize ) )

!  symmetric linear system components

      CALL SLS_matlab_inform_get( FDC_inform%SLS_inform,                       &
                                  FDC_pointer%SLS_pointer )

!  unsymmetric linear system components

      CALL ULS_matlab_inform_get( FDC_inform%ULS_inform,                       &
                                  FDC_pointer%ULS_pointer )

      RETURN

!  End of subroutine FDC_matlab_inform_get

      END SUBROUTINE FDC_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ F D C _ T Y P E S   M O D U L E  -*-*-*-*-

    END MODULE GALAHAD_FDC_MATLAB_TYPES



