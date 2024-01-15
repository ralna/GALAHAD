#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.3 - 2023-12-30 AT 15:20 GMT.

!-*-*-*-  G A L A H A D _ S L L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. July 13th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SLLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SLLS

      USE GALAHAD_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_SLLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SLLS_matlab_control_set, SLLS_matlab_control_get,              &
                SLLS_matlab_inform_create, SLLS_matlab_inform_get

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

      TYPE, PUBLIC :: SLLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
      END TYPE

      TYPE, PUBLIC :: SLLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, factorization_status
        mwPointer :: iter, cg_iter, obj, norm_pg
        TYPE ( SLLS_time_pointer_type ) :: time_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE
    CONTAINS

!-*-  B L L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE SLLS_matlab_control_set( ps, SLLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SLLS

!  Arguments

!  ps - given pointer to the structure
!  SLLS_control - SLLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SLLS_control_type ) :: SLLS_control

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
                                 pc, SLLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SLLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SLLS_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, SLLS_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, SLLS_control%stop_print )
        CASE( 'print_gap' )
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, SLLS_control%print_gap )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, SLLS_control%maxit )
        CASE( 'cold_start' )
          CALL MATLAB_get_value( ps, 'cold_start',                             &
                                 pc, SLLS_control%cold_start )
        CASE( 'preconditioner' )
          CALL MATLAB_get_value( ps, 'preconditioner',                         &
                                 pc, SLLS_control%preconditioner )
        CASE( 'change_max' )
          CALL MATLAB_get_value( ps, 'change_max',                             &
                                 pc, SLLS_control%change_max )
        CASE( 'cg_maxit' )
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, SLLS_control%cg_maxit )
        CASE( 'arcsearch_max_steps' )
          CALL MATLAB_get_value( ps, 'arcsearch_max_steps',                    &
                                 pc, SLLS_control%arcsearch_max_steps )
        CASE( 'weight' )
          CALL MATLAB_get_value( ps, 'weight',                                 &
                                 pc, SLLS_control%weight )
        CASE( 'stop_d' )
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, SLLS_control%stop_d )
        CASE( 'stop_cg_relative' )
          CALL MATLAB_get_value( ps, 'stop_cg_relative',                       &
                                 pc, SLLS_control%stop_cg_relative )
        CASE( 'stop_cg_absolute' )
          CALL MATLAB_get_value( ps, 'stop_cg_absolute',                       &
                                 pc, SLLS_control%stop_cg_absolute )
        CASE( 'alpha_max' )
          CALL MATLAB_get_value( ps, 'alpha_max',                              &
                                 pc, SLLS_control%alpha_max )
        CASE( 'alpha_initial' )
          CALL MATLAB_get_value( ps, 'alpha_initial',                          &
                                 pc, SLLS_control%alpha_initial )
        CASE( 'alpha_reduction' )
          CALL MATLAB_get_value( ps, 'alpha_reduction',                        &
                                 pc, SLLS_control%alpha_reduction )
        CASE( 'arcsearch_acceptance_tol' )
          CALL MATLAB_get_value( ps, 'arcsearch_acceptance_tol',               &
                                 pc, SLLS_control%arcsearch_acceptance_tol )
        CASE( 'stabilisation_weight' )
          CALL MATLAB_get_value( ps, 'stabilisation_weight',                   &
                                 pc, SLLS_control%stabilisation_weight )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, SLLS_control%cpu_time_limit )
        CASE( 'direct_subproblem_solve' )
          CALL MATLAB_get_value( ps, 'direct_subproblem_solve',                &
                                 pc, SLLS_control%direct_subproblem_solve )
        CASE( 'exact_arc_search' )
          CALL MATLAB_get_value( ps, 'exact_arc_search',                       &
                                 pc, SLLS_control%exact_arc_search )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SLLS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, SLLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SLLS_control%prefix, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, SLLS_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SLLS_matlab_control_set

      END SUBROUTINE SLLS_matlab_control_set

!-*-  B L L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE SLLS_matlab_control_get( struct, SLLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SLLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SLLS_control - SLLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLLS_control_type ) :: SLLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 28
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'cold_start                     ', &
         'preconditioner                 ', 'change_max                     ', &
         'cg_maxit                       ', 'arcsearch_max_steps            ', &
         'weight                         ', 'stop_d                         ', &
         'stop_cg_relative               ', 'stop_cg_absolute               ', &
         'alpha_max                      ', 'alpha_initial                  ', &
         'alpha_reduction                ', 'arcsearch_acceptance_tol       ', &
         'stabilisation_weight           ', 'cpu_time_limit                 ', &
         'direct_subproblem_solve        ', 'exact_arc_search               ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         ', 'SBLS_control                   ' /)

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
                                  SLLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SLLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SLLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  SLLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  SLLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  SLLS_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  SLLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'cold_start',                       &
                                  SLLS_control%cold_start )
      CALL MATLAB_fill_component( pointer, 'preconditioner',                   &
                                  SLLS_control%preconditioner )
      CALL MATLAB_fill_component( pointer, 'change_max',                       &
                                  SLLS_control%change_max )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  SLLS_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'arcsearch_max_steps',              &
                                  SLLS_control%arcsearch_max_steps )
      CALL MATLAB_fill_component( pointer, 'weight',                           &
                                  SLLS_control%weight )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  SLLS_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_cg_relative',                 &
                                  SLLS_control%stop_cg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_cg_absolute',                 &
                                  SLLS_control%stop_cg_absolute )
      CALL MATLAB_fill_component( pointer, 'alpha_max',                        &
                                  SLLS_control%alpha_max )
      CALL MATLAB_fill_component( pointer, 'alpha_initial',                    &
                                  SLLS_control%alpha_initial )
      CALL MATLAB_fill_component( pointer, 'alpha_reduction',                  &
                                  SLLS_control%alpha_reduction )
      CALL MATLAB_fill_component( pointer, 'arcsearch_acceptance_tol',         &
                                  SLLS_control%arcsearch_acceptance_tol )
      CALL MATLAB_fill_component( pointer, 'stabilisation_weight',             &
                                  SLLS_control%stabilisation_weight )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  SLLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'direct_subproblem_solve',          &
                                  SLLS_control%direct_subproblem_solve )
      CALL MATLAB_fill_component( pointer, 'exact_arc_search',                 &
                                  SLLS_control%exact_arc_search )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SLLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  SLLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SLLS_control%prefix )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, SLLS_control%SBLS_control,        &
                                    'SBLS_control' )

      RETURN

!  End of subroutine SLLS_matlab_control_get

      END SUBROUTINE SLLS_matlab_control_get

!-*- B L L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE SLLS_matlab_inform_create( struct, SLLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SLLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SLLS_pointer - SLLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SLLS_pointer_type ) :: SLLS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_status ',                   &
           'iter                 ', 'cg_iter              ',                   &
           'obj                  ', 'norm_pg              ',                   &
           'time                 ', 'SBLS_inform          ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 4
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'analyse              ',                   &
           'factorize            ', 'solve                '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SLLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SLLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SLLS_pointer%pointer,              &
        'status', SLLS_pointer%status )
      CALL MATLAB_create_integer_component( SLLS_pointer%pointer,              &
         'alloc_status', SLLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( SLLS_pointer%pointer,                 &
        'bad_alloc', SLLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SLLS_pointer%pointer,              &
        'factorization_status', SLLS_pointer%factorization_status )
      CALL MATLAB_create_integer_component( SLLS_pointer%pointer,              &
        'iter', SLLS_pointer%iter )
      CALL MATLAB_create_integer_component( SLLS_pointer%pointer,              &
        'cg_iter', SLLS_pointer%cg_iter )
      CALL MATLAB_create_real_component( SLLS_pointer%pointer,                 &
        'obj', SLLS_pointer%obj )
      CALL MATLAB_create_real_component( SLLS_pointer%pointer,                 &
         'norm_pg', SLLS_pointer%norm_pg )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( SLLS_pointer%pointer,                   &
        'time', SLLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( SLLS_pointer%time_pointer%pointer,    &
        'total', SLLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( SLLS_pointer%time_pointer%pointer,    &
        'analyse', SLLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( SLLS_pointer%time_pointer%pointer,    &
        'factorize', SLLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( SLLS_pointer%time_pointer%pointer,    &
        'solve', SLLS_pointer%time_pointer%solve )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( SLLS_pointer%pointer,                    &
                                      SLLS_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine SLLS_matlab_inform_create

      END SUBROUTINE SLLS_matlab_inform_create

!-*-  B L L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE SLLS_matlab_inform_get( SLLS_inform, SLLS_pointer )

!  --------------------------------------------------------------

!  Set SLLS_inform values from matlab pointers

!  Arguments

!  SLLS_inform - SLLS inform structure
!  SLLS_pointer - SLLS pointer structure

!  --------------------------------------------------------------

      TYPE ( SLLS_inform_type ) :: SLLS_inform
      TYPE ( SLLS_pointer_type ) :: SLLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SLLS_inform%status,                             &
                               mxGetPr( SLLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SLLS_inform%alloc_status,                       &
                               mxGetPr( SLLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SLLS_pointer%pointer,                           &
                               'bad_alloc', SLLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SLLS_inform%factorization_status,               &
                               mxGetPr( SLLS_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( SLLS_inform%iter,                               &
                               mxGetPr( SLLS_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( SLLS_inform%cg_iter,                            &
                               mxGetPr( SLLS_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( SLLS_inform%obj,                                &
                               mxGetPr( SLLS_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( SLLS_inform%norm_pg,                            &
                               mxGetPr( SLLS_pointer%norm_pg ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( SLLS_inform%time%total, wp ),             &
                               mxGetPr( SLLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLS_inform%time%analyse, wp ),           &
                               mxGetPr( SLLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLS_inform%time%factorize, wp ),         &
                               mxGetPr( SLLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( SLLS_inform%time%solve, wp ),             &
                               mxGetPr( SLLS_pointer%time_pointer%solve ) )

!  positive-definite linear solvers

      CALL SBLS_matlab_inform_get( SLLS_inform%SBLS_inform,                    &
                                   SLLS_pointer%SBLS_pointer )

      RETURN

!  End of subroutine SLLS_matlab_inform_get

      END SUBROUTINE SLLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B L L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SLLS_MATLAB_TYPES
