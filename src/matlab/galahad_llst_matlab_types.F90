#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.6 - 01/03/2014 AT 16:20 GMT.

!-*-*-*-  G A L A H A D _ L L S T _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.6. March 1st, 2014

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LLST_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LLST

      USE GALAHAD_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_IR_MATLAB_TYPES
      USE GALAHAD_LLST_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LLST_matlab_control_set, LLST_matlab_control_get,              &
                LLST_matlab_inform_create, LLST_matlab_inform_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: slen = 30
      INTEGER, PARAMETER :: history_max = 100

!--------------------------
!  Derived type definitions
!--------------------------

      TYPE, PUBLIC :: LLST_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, assemble, analyse, factorize, solve
        mwPointer :: clock_total, clock_assemble
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: LLST_history_pointer_type
        mwPointer :: pointer
        mwPointer :: lambda, x_norm
      END TYPE

      TYPE, PUBLIC :: LLST_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: factorizations, len_history
        mwPointer :: r_norm, x_norm, multiplier
        mwPointer :: time, history
        TYPE ( LLST_time_pointer_type ) :: time_pointer
        TYPE ( LLST_history_pointer_type ) :: history_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
        TYPE ( IR_pointer_type ) :: IR_pointer
      END TYPE 

    CONTAINS

!-*-*-  T R S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-*-

      SUBROUTINE LLST_matlab_control_set( ps, LLST_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LLST

!  Arguments

!  ps - given pointer to the structure
!  SLS_control - SLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LLST_control_type ) :: LLST_control

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
                                 pc, LLST_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LLST_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LLST_control%print_level )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, LLST_control%new_a )
        CASE( 'new_s' )
          CALL MATLAB_get_value( ps, 'new_s',                                  &
                                 pc, LLST_control%new_s )
        CASE( 'max_factorizations' )
          CALL MATLAB_get_value( ps, 'max_factorizations',                     &
                                 pc, LLST_control%max_factorizations )
        CASE( 'taylor_max_degree' )
          CALL MATLAB_get_value( ps, 'taylor_max_degree',                      &
                                 pc, LLST_control%taylor_max_degree )
        CASE( 'initial_multiplier' )
          CALL MATLAB_get_value( ps, 'initial_multiplier',                     &
                                 pc, LLST_control%initial_multiplier )
        CASE( 'lower' )
          CALL MATLAB_get_value( ps, 'lower',                                  &
                                 pc, LLST_control%lower )
        CASE( 'upper' )
          CALL MATLAB_get_value( ps, 'upper',                                  &
                                 pc, LLST_control%upper )
        CASE( 'stop_normal' )
          CALL MATLAB_get_value( ps, 'stop_normal',                            &
                                 pc, LLST_control%stop_normal )
        CASE( 'equality_problem' )
          CALL MATLAB_get_value( ps, 'equality_problem',                       &
                                 pc, LLST_control%equality_problem )
        CASE( 'use_initial_multiplier' )
          CALL MATLAB_get_value( ps, 'use_initial_multiplier',                 &
                                 pc, LLST_control%use_initial_multiplier )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LLST_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LLST_control%deallocate_error_fatal )
        CASE( 'definite_linear_solver' )
          CALL galmxGetCharacter( ps, 'definite_linear_solver',                &
                                  pc, LLST_control%definite_linear_solver, len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LLST_control%prefix, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, LLST_control%SBLS_control, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, LLST_control%SLS_control, len )
        CASE( 'IR_control' )
          pc = mxGetField( ps, 1_mwi_, 'IR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component IR_control must be a structure' )
          CALL IR_matlab_control_set( pc, LLST_control%IR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LLST_matlab_control_set

      END SUBROUTINE LLST_matlab_control_set

!-*-  L L S T _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LLST_matlab_control_get( struct, LLST_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LLST

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LLST_control - LLST control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LLST_control_type ) :: LLST_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 20
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'new_a                          ', &
         'new_s                          ', 'max_factorizations             ', &
         'taylor_max_degree              ', 'initial_multiplier             ', &
         'lower                          ', 'upper                          ', &
         'stop_normal                    ', 'equality_problem               ', &
         'use_initial_multiplier         ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'definite_linear_solver         ', &
         'prefix                         ', 'SBLS_control                   ', &
         'SLS_control                    ', 'IR_control                     ' /)

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
                                  LLST_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LLST_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LLST_control%print_level )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  LLST_control%new_a )
      CALL MATLAB_fill_component( pointer, 'new_s',                            &
                                  LLST_control%new_s )
      CALL MATLAB_fill_component( pointer, 'max_factorizations',               &
                                  LLST_control%max_factorizations )
      CALL MATLAB_fill_component( pointer, 'taylor_max_degree',                &
                                  LLST_control%taylor_max_degree )
      CALL MATLAB_fill_component( pointer, 'initial_multiplier',               &
                                  LLST_control%initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'lower',                            &
                                  LLST_control%lower )
      CALL MATLAB_fill_component( pointer, 'upper',                            &
                                  LLST_control%upper )
      CALL MATLAB_fill_component( pointer, 'stop_normal',                      &
                                  LLST_control%stop_normal )
      CALL MATLAB_fill_component( pointer, 'equality_problem',                 &
                                  LLST_control%equality_problem )
      CALL MATLAB_fill_component( pointer, 'use_initial_multiplier',           &
                                  LLST_control%use_initial_multiplier )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LLST_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LLST_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'definite_linear_solver',           &
                                  LLST_control%definite_linear_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LLST_control%prefix )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, LLST_control%SBLS_control,        &
                                   'SBLS_control' )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, LLST_control%SLS_control,          &
                                   'SLS_control' )

!  create the components of sub-structure IR_control

      CALL IR_matlab_control_get( pointer, LLST_control%IR_control,            &
                                  'IR_control' )

      RETURN

!  End of subroutine LLST_matlab_control_get

      END SUBROUTINE LLST_matlab_control_get

!-   L L S T _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   -

      SUBROUTINE LLST_matlab_inform_create( struct, LLST_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LLST_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LLST_pointer - LLST pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LLST_pointer_type ) :: LLST_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 14
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorizations       ',                   &
           'len_history          ', 'r_norm               ',                   &
           'x_norm               ', 'multiplier           ',                   &
           'pole                 ', 'time                 ',                   &
           'history              ', 'SBLS_inform          ',                   &
           'SLS_inform           ', 'IR_inform            ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'assemble             ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_assemble       ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          ' /)
      INTEGER * 4, PARAMETER :: h_ninform = 2
      CHARACTER ( LEN = 21 ), PARAMETER :: h_finform( h_ninform ) = (/         &
           'lambda               ', 'x_norm               ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LLST_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LLST_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LLST_pointer%pointer,              &
        'status', LLST_pointer%status )
      CALL MATLAB_create_integer_component( LLST_pointer%pointer,              &
         'alloc_status', LLST_pointer%alloc_status )
      CALL MATLAB_create_char_component( LLST_pointer%pointer,                 &
        'bad_alloc', LLST_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LLST_pointer%pointer,              &
        'factorizations', LLST_pointer%factorizations )
      CALL MATLAB_create_integer_component( LLST_pointer%pointer,              &
        'len_history', LLST_pointer%len_history )
      CALL MATLAB_create_real_component( LLST_pointer%pointer,                 &
        'r_norm', LLST_pointer%r_norm )
      CALL MATLAB_create_real_component( LLST_pointer%pointer,                 &
        'x_norm', LLST_pointer%x_norm )
      CALL MATLAB_create_real_component( LLST_pointer%pointer,                 &
        'multiplier', LLST_pointer%multiplier )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( LLST_pointer%pointer,                   &
        'time', LLST_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'total', LLST_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'assemble', LLST_pointer%time_pointer%assemble )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'analyse', LLST_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'factorize', LLST_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'solve', LLST_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'clock_total', LLST_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'clock_assemble', LLST_pointer%time_pointer%clock_assemble )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'clock_analyse', LLST_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'clock_factorize', LLST_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( LLST_pointer%time_pointer%pointer,    &
        'clock_solve', LLST_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure history

      CALL MATLAB_create_substructure( LLST_pointer%pointer,                   &
        'history', LLST_pointer%history_pointer%pointer, h_ninform, h_finform )
      CALL MATLAB_create_real_component(                                       &
        LLST_pointer%history_pointer%pointer,                                  &
        'lambda', history_max, LLST_pointer%history_pointer%lambda )
      CALL MATLAB_create_real_component(                                       &
        LLST_pointer%history_pointer%pointer,                                  &
        'x_norm', history_max, LLST_pointer%history_pointer%x_norm )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( LLST_pointer%pointer,                    &
                                      LLST_pointer%SBLS_pointer, 'SBLS_inform' )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( LLST_pointer%pointer,                     &
                                     LLST_pointer%SLS_pointer, 'SLS_inform' )

!  Define the components of sub-structure IR_inform

      CALL IR_matlab_inform_create( LLST_pointer%pointer,                      &
                                    LLST_pointer%IR_pointer, 'IR_inform' )

      RETURN

!  End of subroutine LLST_matlab_inform_create

      END SUBROUTINE LLST_matlab_inform_create

!-*-  L L S T _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE LLST_matlab_inform_get( LLST_inform, LLST_pointer )

!  --------------------------------------------------------------

!  Set LLST_inform values from matlab pointers

!  Arguments

!  LLST_inform - LLST inform structure
!  LLST_pointer - LLST pointer structure

!  --------------------------------------------------------------

      TYPE ( LLST_inform_type ) :: LLST_inform
      TYPE ( LLST_pointer_type ) :: LLST_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LLST_inform%status,                             &
                               mxGetPr( LLST_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LLST_inform%alloc_status,                       &
                               mxGetPr( LLST_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LLST_pointer%pointer,                           &
                               'bad_alloc', LLST_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LLST_inform%factorizations,                     &
                               mxGetPr( LLST_pointer%factorizations ) )
      CALL MATLAB_copy_to_ptr( LLST_inform%len_history,                        &
                               mxGetPr( LLST_pointer%len_history ) )
      CALL MATLAB_copy_to_ptr( LLST_inform%r_norm,                             &
                               mxGetPr( LLST_pointer%r_norm ) )
      CALL MATLAB_copy_to_ptr( LLST_inform%x_norm,                             &
                               mxGetPr( LLST_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( LLST_inform%multiplier,                         &
                               mxGetPr( LLST_pointer%multiplier ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%total, wp ),             &
                           mxGetPr( LLST_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%assemble, wp ),          &
                           mxGetPr( LLST_pointer%time_pointer%assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%analyse, wp ),           &
                           mxGetPr( LLST_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%factorize, wp ),         &
                           mxGetPr( LLST_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%solve, wp ),             &
                           mxGetPr( LLST_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%clock_total, wp ),       &
                           mxGetPr( LLST_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%clock_assemble, wp ),    &
                           mxGetPr( LLST_pointer%time_pointer%clock_assemble ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%clock_analyse, wp ),     &
                           mxGetPr( LLST_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%clock_factorize, wp ),   &
                          mxGetPr( LLST_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LLST_inform%time%clock_solve, wp ),       &
                           mxGetPr( LLST_pointer%time_pointer%clock_solve ) )

!  history components

      CALL MATLAB_copy_to_ptr( LLST_inform%history%lambda,                     &
                               mxGetPr( LLST_pointer%history_pointer%lambda ), &
                               history_max )
      CALL MATLAB_copy_to_ptr( LLST_inform%history%x_norm,                     &
                               mxGetPr( LLST_pointer%history_pointer%x_norm ), &
                               history_max )

!  structured linear system components

      CALL SBLS_matlab_inform_get( LLST_inform%SBLS_inform,                    &
                                   LLST_pointer%SBLS_pointer )

!  linear system components

      CALL SLS_matlab_inform_get( LLST_inform%SLS_inform,                      &
                                  LLST_pointer%SLS_pointer )

!  iterative_refinement components

      CALL IR_matlab_inform_get( LLST_inform%IR_inform,                        &
                                 LLST_pointer%IR_pointer )

      RETURN

!  End of subroutine LLST_matlab_inform_get

      END SUBROUTINE LLST_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L L S T _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LLST_MATLAB_TYPES
