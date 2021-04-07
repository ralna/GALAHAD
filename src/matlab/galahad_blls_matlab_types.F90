#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.3 - 12/12/2020 AT 16:15 GMT.

!-*-*-*-  G A L A H A D _ B L L S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. December 12th, 2020

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BLLS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BLLS

      USE GALAHAD_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_BLLS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BLLS_matlab_control_set, BLLS_matlab_control_get,              &
                BLLS_matlab_inform_create, BLLS_matlab_inform_get

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

      TYPE, PUBLIC :: BLLS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
      END TYPE 

      TYPE, PUBLIC :: BLLS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, factorization_status
        mwPointer :: iter, cg_iter, obj, norm_pg
        TYPE ( BLLS_time_pointer_type ) :: time_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE 
    CONTAINS

!-*-  B L L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BLLS_matlab_control_set( ps, BLLS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BLLS

!  Arguments

!  ps - given pointer to the structure
!  BLLS_control - BLLS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BLLS_control_type ) :: BLLS_control

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
                                 pc, BLLS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BLLS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BLLS_control%print_level )
        CASE( 'start_print' )                                                  
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BLLS_control%start_print )
        CASE( 'stop_print' )                                                    
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BLLS_control%stop_print )
        CASE( 'print_gap' )                                                    
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, BLLS_control%print_gap )
        CASE( 'maxit' )                 
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, BLLS_control%maxit )
        CASE( 'cold_start' )                                         
          CALL MATLAB_get_value( ps, 'cold_start',                             &
                                 pc, BLLS_control%cold_start )
        CASE( 'preconditioner' )                                         
          CALL MATLAB_get_value( ps, 'preconditioner',                         &
                                 pc, BLLS_control%preconditioner )
        CASE( 'change_max' )                                         
          CALL MATLAB_get_value( ps, 'change_max',                             &
                                 pc, BLLS_control%change_max )
        CASE( 'cg_maxit' )                                                    
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, BLLS_control%cg_maxit )
        CASE( 'arcsearch_max_steps' )                                         
          CALL MATLAB_get_value( ps, 'arcsearch_max_steps',                    &
                                 pc, BLLS_control%arcsearch_max_steps )
        CASE( 'weight' )                                                     
          CALL MATLAB_get_value( ps, 'weight',                                 &
                                 pc, BLLS_control%weight )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BLLS_control%infinity )
        CASE( 'stop_d' )                                                        
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, BLLS_control%stop_d )
        CASE( 'identical_bounds_tol' )                                         
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, BLLS_control%identical_bounds_tol )
        CASE( 'stop_cg_relative' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_relative',                       &
                                 pc, BLLS_control%stop_cg_relative )
        CASE( 'stop_cg_absolute' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_absolute',                       &
                                 pc, BLLS_control%stop_cg_absolute )
        CASE( 'alpha_max' )                                         
          CALL MATLAB_get_value( ps, 'alpha_max',                              &
                                 pc, BLLS_control%alpha_max )
        CASE( 'alpha_initial' )                                         
          CALL MATLAB_get_value( ps, 'alpha_initial',                          &
                                 pc, BLLS_control%alpha_initial )
        CASE( 'alpha_reduction' )                                         
          CALL MATLAB_get_value( ps, 'alpha_reduction',                        &
                                 pc, BLLS_control%alpha_reduction )
        CASE( 'arcsearch_acceptance_tol' )                                     
          CALL MATLAB_get_value( ps, 'arcsearch_acceptance_tol',               &
                                 pc, BLLS_control%arcsearch_acceptance_tol )
        CASE( 'stabilisation_weight' )                                         
          CALL MATLAB_get_value( ps, 'stabilisation_weight',                   &
                                 pc, BLLS_control%stabilisation_weight )
        CASE( 'cpu_time_limit' )                                               
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BLLS_control%cpu_time_limit )
        CASE( 'direct_subproblem_solve' )       
          CALL MATLAB_get_value( ps, 'direct_subproblem_solve',                &
                                 pc, BLLS_control%direct_subproblem_solve )
        CASE( 'exact_arc_search' )                                         
          CALL MATLAB_get_value( ps, 'exact_arc_search',                       &
                                 pc, BLLS_control%exact_arc_search )
        CASE( 'advance' )                                         
          CALL MATLAB_get_value( ps, 'advance',                                &
                                 pc, BLLS_control%advance )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BLLS_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BLLS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BLLS_control%prefix, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, BLLS_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BLLS_matlab_control_set

      END SUBROUTINE BLLS_matlab_control_set

!-*-  B L L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BLLS_matlab_control_get( struct, BLLS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BLLS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BLLS_control - BLLS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BLLS_control_type ) :: BLLS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 31
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'maxit                          ', 'cold_start                     ', &
         'preconditioner                 ', 'change_max                     ', &
         'cg_maxit                       ', 'arcsearch_max_steps            ', &
         'weight                         ',                                    &
         'infinity                       ', 'stop_d                         ', &
         'identical_bounds_tol           ', 'stop_cg_relative               ', &
         'stop_cg_absolute               ', 'alpha_max                      ', &
         'alpha_initial                  ', 'alpha_reduction                ', &
         'arcsearch_acceptance_tol       ', 'stabilisation_weight           ', &
         'cpu_time_limit                 ', 'direct_subproblem_solve        ', &
         'exact_arc_search               ', 'advance                        ', &
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
                                  BLLS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BLLS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BLLS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BLLS_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BLLS_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  BLLS_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  BLLS_control%maxit )
      CALL MATLAB_fill_component( pointer, 'cold_start',                       &
                                  BLLS_control%cold_start )
      CALL MATLAB_fill_component( pointer, 'preconditioner',                   &
                                  BLLS_control%preconditioner )
      CALL MATLAB_fill_component( pointer, 'change_max',                       &
                                  BLLS_control%change_max )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  BLLS_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'arcsearch_max_steps',              &
                                  BLLS_control%arcsearch_max_steps )
      CALL MATLAB_fill_component( pointer, 'weight',                           &
                                  BLLS_control%weight )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BLLS_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  BLLS_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  BLLS_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'stop_cg_relative',                 &
                                  BLLS_control%stop_cg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_cg_absolute',                 &
                                  BLLS_control%stop_cg_absolute )
      CALL MATLAB_fill_component( pointer, 'alpha_max',                        &
                                  BLLS_control%alpha_max )
      CALL MATLAB_fill_component( pointer, 'alpha_initial',                    &
                                  BLLS_control%alpha_initial )
      CALL MATLAB_fill_component( pointer, 'alpha_reduction',                  &
                                  BLLS_control%alpha_reduction )
      CALL MATLAB_fill_component( pointer, 'arcsearch_acceptance_tol',         &
                                  BLLS_control%arcsearch_acceptance_tol )
      CALL MATLAB_fill_component( pointer, 'stabilisation_weight',             &
                                  BLLS_control%stabilisation_weight )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BLLS_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'direct_subproblem_solve',          &
                                  BLLS_control%direct_subproblem_solve )
      CALL MATLAB_fill_component( pointer, 'exact_arc_search',                 &
                                  BLLS_control%exact_arc_search )
      CALL MATLAB_fill_component( pointer, 'advance',                          &
                                  BLLS_control%advance )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BLLS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BLLS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BLLS_control%prefix )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, BLLS_control%SBLS_control,        &
                                    'SBLS_control' )

      RETURN

!  End of subroutine BLLS_matlab_control_get

      END SUBROUTINE BLLS_matlab_control_get

!-*- B L L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE BLLS_matlab_inform_create( struct, BLLS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BLLS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BLLS_pointer - BLLS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BLLS_pointer_type ) :: BLLS_pointer
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
        CALL MATLAB_create_substructure( struct, name, BLLS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BLLS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BLLS_pointer%pointer,              &
        'status', BLLS_pointer%status )
      CALL MATLAB_create_integer_component( BLLS_pointer%pointer,              &
         'alloc_status', BLLS_pointer%alloc_status )
      CALL MATLAB_create_char_component( BLLS_pointer%pointer,                 &
        'bad_alloc', BLLS_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BLLS_pointer%pointer,              &
        'factorization_status', BLLS_pointer%factorization_status )
      CALL MATLAB_create_integer_component( BLLS_pointer%pointer,              &
        'iter', BLLS_pointer%iter )
      CALL MATLAB_create_integer_component( BLLS_pointer%pointer,              &
        'cg_iter', BLLS_pointer%cg_iter )
      CALL MATLAB_create_real_component( BLLS_pointer%pointer,                 &
        'obj', BLLS_pointer%obj )
      CALL MATLAB_create_real_component( BLLS_pointer%pointer,                 &
         'norm_pg', BLLS_pointer%norm_pg )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BLLS_pointer%pointer,                   &
        'time', BLLS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BLLS_pointer%time_pointer%pointer,    &
        'total', BLLS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BLLS_pointer%time_pointer%pointer,    &
        'analyse', BLLS_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( BLLS_pointer%time_pointer%pointer,    &
        'factorize', BLLS_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( BLLS_pointer%time_pointer%pointer,    &
        'solve', BLLS_pointer%time_pointer%solve )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( BLLS_pointer%pointer,                    &
                                      BLLS_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine BLLS_matlab_inform_create

      END SUBROUTINE BLLS_matlab_inform_create

!-*-  B L L S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE BLLS_matlab_inform_get( BLLS_inform, BLLS_pointer )

!  --------------------------------------------------------------

!  Set BLLS_inform values from matlab pointers

!  Arguments

!  BLLS_inform - BLLS inform structure
!  BLLS_pointer - BLLS pointer structure

!  --------------------------------------------------------------

      TYPE ( BLLS_inform_type ) :: BLLS_inform
      TYPE ( BLLS_pointer_type ) :: BLLS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BLLS_inform%status,                             &
                               mxGetPr( BLLS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BLLS_inform%alloc_status,                       &
                               mxGetPr( BLLS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BLLS_pointer%pointer,                           &
                               'bad_alloc', BLLS_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BLLS_inform%factorization_status,               &
                               mxGetPr( BLLS_pointer%factorization_status ) )   
      CALL MATLAB_copy_to_ptr( BLLS_inform%iter,                               &
                               mxGetPr( BLLS_pointer%iter ) )             
      CALL MATLAB_copy_to_ptr( BLLS_inform%cg_iter,                            &
                               mxGetPr( BLLS_pointer%cg_iter ) )             
      CALL MATLAB_copy_to_ptr( BLLS_inform%obj,                                &
                               mxGetPr( BLLS_pointer%obj ) )                    
      CALL MATLAB_copy_to_ptr( BLLS_inform%norm_pg,                            &
                               mxGetPr( BLLS_pointer%norm_pg ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( BLLS_inform%time%total, wp ),             &
                               mxGetPr( BLLS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLS_inform%time%analyse, wp ),           &
                               mxGetPr( BLLS_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLS_inform%time%factorize, wp ),         &
                               mxGetPr( BLLS_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BLLS_inform%time%solve, wp ),             &
                               mxGetPr( BLLS_pointer%time_pointer%solve ) )

!  positive-definite linear solvers

      CALL SBLS_matlab_inform_get( BLLS_inform%SBLS_inform,                    &
                                   BLLS_pointer%SBLS_pointer )

      RETURN

!  End of subroutine BLLS_matlab_inform_get

      END SUBROUTINE BLLS_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B L L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_BLLS_MATLAB_TYPES
