#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 15:45 GMT.

!-*-*-*-  G A L A H A D _ B Q P B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BQPB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BQPB

      USE GALAHAD_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_BQPB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BQPB_matlab_control_set, BQPB_matlab_control_get,              &
                BQPB_matlab_inform_create, BQPB_matlab_inform_get

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

      TYPE, PUBLIC :: BQPB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
      END TYPE 

      TYPE, PUBLIC :: BQPB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, factorization_status
        mwPointer :: iter, cg_iter, obj, norm_pg, slknes
        TYPE ( BQPB_time_pointer_type ) :: time_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE 
    CONTAINS

!-*-  B Q P B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BQPB_matlab_control_set( ps, BQPB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BQPB

!  Arguments

!  ps - given pointer to the structure
!  BQPB_control - BQPB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BQPB_control_type ) :: BQPB_control

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
                                 pc, BQPB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BQPB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BQPB_control%print_level )
        CASE( 'start_print' )                                                  
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BQPB_control%start_print )
        CASE( 'stop_print' )                                                    
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BQPB_control%stop_print )
        CASE( 'print_gap' )                                                    
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, BQPB_control%print_gap )
        CASE( 'ratio_cg_vs_sd' )                 
          CALL MATLAB_get_value( ps, 'ratio_cg_vs_sd',                         &
                                 pc, BQPB_control%ratio_cg_vs_sd )
        CASE( 'cg_maxit' )                                                    
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, BQPB_control%cg_maxit )
        CASE( 'stepsize_strategy' )                                             
          CALL MATLAB_get_value( ps, 'stepsize_strategy',                      &
                                 pc, BQPB_control%stepsize_strategy )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BQPB_control%infinity )
        CASE( 'stop_p' )
          CALL MATLAB_get_value( ps, 'stop_p',                                 &
                                 pc, BQPB_control%stop_p )
        CASE( 'stop_d' )                                                        
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, BQPB_control%stop_d )
        CASE( 'stop_c' )                                                        
          CALL MATLAB_get_value( ps, 'stop_c',                                 &
                                 pc, BQPB_control%stop_c )
        CASE( 'identical_bounds_tol' )                                         
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, BQPB_control%identical_bounds_tol )
        CASE( 'mu_zero' )                                                    
          CALL MATLAB_get_value( ps, 'mu_zero',                                &
                                 pc, BQPB_control%mu_zero )
        CASE( 'pr_feas' )                                                    
          CALL MATLAB_get_value( ps, 'pr_feas',                                &
                                 pc, BQPB_control%pr_feas )
        CASE( 'du_feas' )                                                    
          CALL MATLAB_get_value( ps, 'du_feas',                                &
                                 pc, BQPB_control%du_feas )
        CASE( 'fraction_to_the_boundary' )                                
          CALL MATLAB_get_value( ps, 'fraction_to_the_boundary',               &
                                 pc, BQPB_control%fraction_to_the_boundary )
        CASE( 'stop_cg_relative' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_relative',                       &
                                 pc, BQPB_control%stop_cg_relative )
        CASE( 'stop_cg_absolute' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_absolute',                       &
                                 pc, BQPB_control%stop_cg_absolute )
        CASE( 'zero_curvature' )                                         
          CALL MATLAB_get_value( ps, 'zero_curvature',                         &
                                 pc, BQPB_control%zero_curvature )
        CASE( 'cpu_time_limit' )                                               
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BQPB_control%cpu_time_limit )
        CASE( 'primal_dual' )                                                 
          CALL MATLAB_get_value( ps, 'primal_dual',                            &
                                 pc, BQPB_control%primal_dual )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BQPB_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BQPB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BQPB_control%prefix, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, BQPB_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BQPB_matlab_control_set

      END SUBROUTINE BQPB_matlab_control_set

!-*-  B Q P B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BQPB_matlab_control_get( struct, BQPB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BQPB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQPB_control - BQPB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQPB_control_type ) :: BQPB_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 27
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'ratio_cg_vs_sd                 ', 'cg_maxit                       ', &
         'stepsize_strategy              ', 'infinity                       ', &
         'stop_p                         ', 'stop_d                         ', &
         'stop_c                         ', 'identical_bounds_tol           ', &
         'mu_zero                        ', 'pr_feas                        ', &
         'du_feas                        ', 'fraction_to_the_boundary       ', &
         'stop_cg_relative               ', 'stop_cg_absolute               ', &
         'zero_curvature                 ', 'cpu_time_limit                 ', &
         'primal_dual                    ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ', &
         'SBLS_control                   '                       /)

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
                                  BQPB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BQPB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BQPB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BQPB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BQPB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  BQPB_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'ratio_cg_vs_sd',                   &
                                  BQPB_control%ratio_cg_vs_sd )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  BQPB_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'stepsize_strategy',                &
                                  BQPB_control%stepsize_strategy )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BQPB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_p',                           &
                                  BQPB_control%stop_p )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  BQPB_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_c',                           &
                                  BQPB_control%stop_c )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  BQPB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'mu_zero',                          &
                                  BQPB_control%mu_zero )
      CALL MATLAB_fill_component( pointer, 'pr_feas',                          &
                                  BQPB_control%pr_feas )
      CALL MATLAB_fill_component( pointer, 'du_feas',                          &
                                  BQPB_control%du_feas )
      CALL MATLAB_fill_component( pointer, 'fraction_to_the_boundary',         &
                                  BQPB_control%fraction_to_the_boundary )
      CALL MATLAB_fill_component( pointer, 'stop_cg_relative',                 &
                                  BQPB_control%stop_cg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_cg_absolute',                 &
                                  BQPB_control%stop_cg_absolute )
      CALL MATLAB_fill_component( pointer, 'zero_curvature',                   &
                                  BQPB_control%zero_curvature )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BQPB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'primal_dual',                      &
                                  BQPB_control%primal_dual )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BQPB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BQPB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BQPB_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SBLS_matlab_control_get( pointer, BQPB_control%SBLS_control,        &
                                   'SBLS_control' )

      RETURN

!  End of subroutine BQPB_matlab_control_get

      END SUBROUTINE BQPB_matlab_control_get

!-*- B Q P B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE BQPB_matlab_inform_create( struct, BQPB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BQPB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQPB_pointer - BQPB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQPB_pointer_type ) :: BQPB_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 11
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_status ',                   &
           'iter                 ', 'cg_iter              ',                   &
           'obj                  ', 'norm_pg              ',                   &
           'slknes               ', 'time                 ',                   &
           'SBLS_inform          ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 4
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'analyse              ',                   &
           'factorize            ', 'solve                '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, BQPB_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BQPB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'status', BQPB_pointer%status )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
         'alloc_status', BQPB_pointer%alloc_status )
      CALL MATLAB_create_char_component( BQPB_pointer%pointer,                 &
        'bad_alloc', BQPB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'factorization_status', BQPB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'iter', BQPB_pointer%iter )
      CALL MATLAB_create_integer_component( BQPB_pointer%pointer,              &
        'cg_iter', BQPB_pointer%cg_iter )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
        'obj', BQPB_pointer%obj )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
         'norm_pg', BQPB_pointer%norm_pg )
      CALL MATLAB_create_real_component( BQPB_pointer%pointer,                 &
         'slknes', BQPB_pointer%slknes )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BQPB_pointer%pointer,                   &
        'time', BQPB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'total', BQPB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'analyse', BQPB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'factorize', BQPB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( BQPB_pointer%time_pointer%pointer,    &
        'solve', BQPB_pointer%time_pointer%solve )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( BQPB_pointer%pointer,                    &
                                      BQPB_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine BQPB_matlab_inform_create

      END SUBROUTINE BQPB_matlab_inform_create

!-*-*-  B Q P B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE BQPB_matlab_inform_get( BQPB_inform, BQPB_pointer )

!  --------------------------------------------------------------

!  Set BQPB_inform values from matlab pointers

!  Arguments

!  BQPB_inform - BQPB inform structure
!  BQPB_pointer - BQPB pointer structure

!  --------------------------------------------------------------

      TYPE ( BQPB_inform_type ) :: BQPB_inform
      TYPE ( BQPB_pointer_type ) :: BQPB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BQPB_inform%status,                             &
                               mxGetPr( BQPB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%alloc_status,                       &
                               mxGetPr( BQPB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BQPB_pointer%pointer,                           &
                               'bad_alloc', BQPB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BQPB_inform%factorization_status,               &
                               mxGetPr( BQPB_pointer%factorization_status ) )   
      CALL MATLAB_copy_to_ptr( BQPB_inform%iter,                               &
                               mxGetPr( BQPB_pointer%iter ) )             
      CALL MATLAB_copy_to_ptr( BQPB_inform%cg_iter,                            &
                               mxGetPr( BQPB_pointer%cg_iter ) )             
      CALL MATLAB_copy_to_ptr( BQPB_inform%obj,                                &
                               mxGetPr( BQPB_pointer%obj ) )                    
      CALL MATLAB_copy_to_ptr( BQPB_inform%norm_pg,                            &
                               mxGetPr( BQPB_pointer%norm_pg ) )
      CALL MATLAB_copy_to_ptr( BQPB_inform%slknes,                             &
                               mxGetPr( BQPB_pointer%slknes ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%total, wp ),             &
                               mxGetPr( BQPB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%analyse, wp ),           &
                               mxGetPr( BQPB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%factorize, wp ),         &
                               mxGetPr( BQPB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BQPB_inform%time%solve, wp ),             &
                               mxGetPr( BQPB_pointer%time_pointer%solve ) )

!  positive-definite linear solvers

      CALL SBLS_matlab_inform_get( BQPB_inform%SBLS_inform,                    &
                                   BQPB_pointer%SBLS_pointer )

      RETURN

!  End of subroutine BQPB_matlab_inform_get

      END SUBROUTINE BQPB_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B Q P B _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_BQPB_MATLAB_TYPES
