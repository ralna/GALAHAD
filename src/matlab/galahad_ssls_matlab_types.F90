#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.3 - 2025-08-09 AT 13:40 GMT.

!-*-*-*-  G A L A H A D _ S S L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.3. August 9th, 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SSLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SSLS

      USE GALAHAD_MATLAB
      USE GALAHAD_ULS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_SSLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SSLS_matlab_control_set, SSLS_matlab_control_get,              &
                SSLS_matlab_inform_create, SSLS_matlab_inform_get

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

      TYPE, PUBLIC :: SSLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
        mwPointer :: clock_total, clock_analyse, clock_factorize, clock_solve
      END TYPE

      TYPE, PUBLIC :: SSLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: rank, rank_def
        mwPointer :: perturbed, iter_pcg, norm_residual
        TYPE ( SSLS_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
      END TYPE
    CONTAINS

!-*-  S S L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE SSLS_matlab_control_set( ps, SSLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SSLS

!  Arguments

!  ps - given pointer to the structure
!  SSLS_control - SSLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SSLS_control_type ) :: SSLS_control

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
                                 pc, SSLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SSLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SSLS_control%print_level )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SSLS_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, SSLS_control%deallocate_error_fatal  )
        CASE( 'symmetric_linear_solver' )
          CALL galmxGetCharacter( ps, 'symmetric_linear_solver',               &
                                  pc, SSLS_control%symmetric_linear_solver,    &
                                  len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SSLS_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, SSLS_control%SLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SSLS_matlab_control_set

      END SUBROUTINE SSLS_matlab_control_set

!-*-  S S L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE SSLS_matlab_control_get( struct, SSLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SSLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SSLS_control - SSLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SSLS_control_type ) :: SSLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 8
      CHARACTER ( LEN = 25 ), PARAMETER :: finform( ninform ) = (/             &
           'error                    ', 'out                      ',           &
           'print_level              ', 'space_critical           ',           &
           'deallocate_error_fatal   ', 'symmetric_linear_solver  ',           &
           'prefix                   ', 'SLS_control              ' /)

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
                                  SSLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SSLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SSLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SSLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  SSLS_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'symmetric_linear_solver',          &
                                  SSLS_control%symmetric_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SSLS_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, SSLS_control%SLS_control,          &
                                  'SLS_control' )

      RETURN

!  End of subroutine SSLS_matlab_control_get

      END SUBROUTINE SSLS_matlab_control_get

!-*- S S L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE SSLS_matlab_inform_create( struct, SSLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SSLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SSLS_pointer - SSLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SSLS_pointer_type ) :: SSLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_integer',                   &
           'factorization_real   ', 'rank                 ',                   &
           'rank_def             ', 'time                 ',                   &
           'SLS_inform           ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'analyse              ',                   &
           'factorize            ', 'solve                ',                   &
           'clock_total          ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SSLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SSLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SSLS_pointer%pointer,              &
        'status', SSLS_pointer%status )
      CALL MATLAB_create_integer_component( SSLS_pointer%pointer,              &
         'alloc_status', SSLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( SSLS_pointer%pointer,                 &
        'bad_alloc', SSLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SSLS_pointer%pointer,              &
         'factorization_integer', SSLS_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( SSLS_pointer%pointer,              &
         'factorization_real', SSLS_pointer%factorization_real )
      CALL MATLAB_create_integer_component( SSLS_pointer%pointer,              &
         'rank', SSLS_pointer%rank )
      CALL MATLAB_create_logical_component( SSLS_pointer%pointer,              &
         'rank_def', SSLS_pointer%rank_def )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( SSLS_pointer%pointer,                   &
        'time', SSLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'total', SSLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'analyse', SSLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'factorize', SSLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'solve', SSLS_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'clock_total', SSLS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'clock_analyse', SSLS_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'clock_factorize', SSLS_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( SSLS_pointer%time_pointer%pointer,    &
        'clock_solve', SSLS_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( SSLS_pointer%pointer,                     &
                                     SSLS_pointer%SLS_pointer, 'SLS_inform' )

      RETURN

!  End of subroutine SSLS_matlab_inform_create

      END SUBROUTINE SSLS_matlab_inform_create

!-*-*  S S L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   *-*-

      SUBROUTINE SSLS_matlab_inform_get( SSLS_inform, SSLS_pointer )

!  --------------------------------------------------------------

!  Set SSLS_inform values from matlab pointers

!  Arguments

!  SSLS_inform - SSLS inform structure
!  SSLS_pointer - SSLS pointer structure

!  --------------------------------------------------------------

      TYPE ( SSLS_inform_type ) :: SSLS_inform
      TYPE ( SSLS_pointer_type ) :: SSLS_pointer

!     INTEGER ::  mexPrintf
!     integer*4 out
!     CHARACTER ( LEN = 200 ) :: str

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SSLS_inform%status,                             &
                               mxGetPr( SSLS_pointer%status ) )
! WRITE( str, "( ' alloc_status'  )" )
! out = mexPrintf( TRIM( str ) // achar(10) )
      CALL MATLAB_copy_to_ptr( SSLS_inform%alloc_status,                       &
                               mxGetPr( SSLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SSLS_pointer%pointer,                           &
                               'bad_alloc', SSLS_inform%bad_alloc )
      CALL galmxCopyLongToPtr( SSLS_inform%factorization_integer,              &
                               mxGetPr( SSLS_pointer%factorization_integer ) )
      CALL galmxCopyLongToPtr( SSLS_inform%factorization_real,                 &
                               mxGetPr( SSLS_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( SSLS_inform%rank,                               &
                               mxGetPr( SSLS_pointer%rank ) )
      CALL MATLAB_copy_to_ptr( SSLS_inform%rank_def,                           &
                               mxGetPr( SSLS_pointer%rank_def ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%total, wp ),             &
                               mxGetPr( SSLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%analyse, wp ),           &
                               mxGetPr( SSLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%factorize, wp ),         &
                               mxGetPr( SSLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%solve, wp ),             &
                               mxGetPr( SSLS_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%clock_total, wp ),       &
                          mxGetPr( SSLS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%clock_analyse, wp ),     &
                          mxGetPr( SSLS_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%clock_factorize, wp ),   &
                          mxGetPr( SSLS_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SSLS_inform%time%clock_solve, wp ),       &
                          mxGetPr( SSLS_pointer%time_pointer%clock_solve ) )

!  symmetric linear system components

      CALL SLS_matlab_inform_get( SSLS_inform%SLS_inform,                      &
                                  SSLS_pointer%SLS_pointer )

      RETURN

!  End of subroutine SSLS_matlab_inform_get

      END SUBROUTINE SSLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ S S L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SSLS_MATLAB_TYPES
