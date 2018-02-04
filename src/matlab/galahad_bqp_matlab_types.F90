#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 15:30 GMT.

!-*-*-*-  G A L A H A D _ B Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BQP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BQP

      USE GALAHAD_MATLAB
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_BQP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BQP_matlab_control_set, BQP_matlab_control_get,                &
                BQP_matlab_inform_create, BQP_matlab_inform_get

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

      TYPE, PUBLIC :: BQP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, analyse, factorize, solve
      END TYPE 

      TYPE, PUBLIC :: BQP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, factorization_status
        mwPointer :: iter, cg_iter, obj, norm_pg
        TYPE ( BQP_time_pointer_type ) :: time_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE 
    CONTAINS

!-*-  B Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BQP_matlab_control_set( ps, BQP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BQP

!  Arguments

!  ps - given pointer to the structure
!  BQP_control - BQP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BQP_control_type ) :: BQP_control

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
                                 pc, BQP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BQP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BQP_control%print_level )
        CASE( 'start_print' )                                                  
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, BQP_control%start_print )
        CASE( 'stop_print' )                                                    
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, BQP_control%stop_print )
        CASE( 'print_gap' )                                                    
          CALL MATLAB_get_value( ps, 'print_gap',                              &
                                 pc, BQP_control%print_gap )
        CASE( 'ratio_cg_vs_sd' )                 
          CALL MATLAB_get_value( ps, 'ratio_cg_vs_sd',                         &
                                 pc, BQP_control%ratio_cg_vs_sd )
        CASE( 'cg_maxit' )                                                    
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, BQP_control%cg_maxit )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, BQP_control%infinity )
        CASE( 'stop_p' )
          CALL MATLAB_get_value( ps, 'stop_p',                                 &
                                 pc, BQP_control%stop_p )
        CASE( 'stop_d' )                                                        
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, BQP_control%stop_d )
        CASE( 'stop_c' )                                                        
          CALL MATLAB_get_value( ps, 'stop_c',                                 &
                                 pc, BQP_control%stop_c )
        CASE( 'identical_bounds_tol' )                                         
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, BQP_control%identical_bounds_tol )
        CASE( 'stop_cg_relative' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_relative',                       &
                                 pc, BQP_control%stop_cg_relative )
        CASE( 'stop_cg_absolute' )                                         
          CALL MATLAB_get_value( ps, 'stop_cg_absolute',                       &
                                 pc, BQP_control%stop_cg_absolute )
        CASE( 'zero_curvature' )                                         
          CALL MATLAB_get_value( ps, 'zero_curvature',                         &
                                 pc, BQP_control%zero_curvature )
        CASE( 'cpu_time_limit' )                                               
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, BQP_control%cpu_time_limit )
        CASE( 'exact_arcsearch' )                                         
          CALL MATLAB_get_value( ps, 'exact_arcsearch',                        &
                                 pc, BQP_control%exact_arcsearch )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BQP_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, BQP_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BQP_control%prefix, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, BQP_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BQP_matlab_control_set

      END SUBROUTINE BQP_matlab_control_set

!-*-  B Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BQP_matlab_control_get( struct, BQP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BQP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQP_control - BQP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQP_control_type ) :: BQP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 22
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'print_gap                      ', &
         'ratio_cg_vs_sd                 ', 'cg_maxit                       ', &
         'infinity                       ', 'stop_p                         ', &
         'stop_d                         ', 'stop_c                         ', &
         'identical_bounds_tol           ', 'stop_cg_relative               ', &
         'stop_cg_absolute               ', 'zero_curvature                 ', &
         'cpu_time_limit                 ', 'exact_arcsearch                ', &
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
                                  BQP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BQP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BQP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  BQP_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  BQP_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'print_gap',                        &
                                  BQP_control%print_gap )
      CALL MATLAB_fill_component( pointer, 'ratio_cg_vs_sd',                   &
                                  BQP_control%ratio_cg_vs_sd )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  BQP_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  BQP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_p',                           &
                                  BQP_control%stop_p )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  BQP_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_c',                           &
                                  BQP_control%stop_c )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  BQP_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'stop_cg_relative',                 &
                                  BQP_control%stop_cg_relative )
      CALL MATLAB_fill_component( pointer, 'stop_cg_absolute',                 &
                                  BQP_control%stop_cg_absolute )
      CALL MATLAB_fill_component( pointer, 'zero_curvature',                   &
                                  BQP_control%zero_curvature )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  BQP_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'exact_arcsearch',                  &
                                  BQP_control%exact_arcsearch )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BQP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  BQP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BQP_control%prefix )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, BQP_control%SBLS_control,         &
                                    'SBLS_control' )

      RETURN

!  End of subroutine BQP_matlab_control_get

      END SUBROUTINE BQP_matlab_control_get

!-*- B Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE BQP_matlab_inform_create( struct, BQP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BQP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BQP_pointer - BQP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BQP_pointer_type ) :: BQP_pointer
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
        CALL MATLAB_create_substructure( struct, name, BQP_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BQP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BQP_pointer%pointer,               &
        'status', BQP_pointer%status )
      CALL MATLAB_create_integer_component( BQP_pointer%pointer,               &
         'alloc_status', BQP_pointer%alloc_status )
      CALL MATLAB_create_char_component( BQP_pointer%pointer,                  &
        'bad_alloc', BQP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BQP_pointer%pointer,               &
        'factorization_status', BQP_pointer%factorization_status )
      CALL MATLAB_create_integer_component( BQP_pointer%pointer,               &
        'iter', BQP_pointer%iter )
      CALL MATLAB_create_integer_component( BQP_pointer%pointer,               &
        'cg_iter', BQP_pointer%cg_iter )
      CALL MATLAB_create_real_component( BQP_pointer%pointer,                  &
        'obj', BQP_pointer%obj )
      CALL MATLAB_create_real_component( BQP_pointer%pointer,                  &
         'norm_pg', BQP_pointer%norm_pg )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( BQP_pointer%pointer,                    &
        'time', BQP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( BQP_pointer%time_pointer%pointer,     &
        'total', BQP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( BQP_pointer%time_pointer%pointer,     &
        'analyse', BQP_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( BQP_pointer%time_pointer%pointer,     &
        'factorize', BQP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( BQP_pointer%time_pointer%pointer,     &
        'solve', BQP_pointer%time_pointer%solve )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( BQP_pointer%pointer,                     &
                                      BQP_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine BQP_matlab_inform_create

      END SUBROUTINE BQP_matlab_inform_create

!-*-*-  B Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE BQP_matlab_inform_get( BQP_inform, BQP_pointer )

!  --------------------------------------------------------------

!  Set BQP_inform values from matlab pointers

!  Arguments

!  BQP_inform - BQP inform structure
!  BQP_pointer - BQP pointer structure

!  --------------------------------------------------------------

      TYPE ( BQP_inform_type ) :: BQP_inform
      TYPE ( BQP_pointer_type ) :: BQP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BQP_inform%status,                              &
                               mxGetPr( BQP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BQP_inform%alloc_status,                        &
                               mxGetPr( BQP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BQP_pointer%pointer,                            &
                               'bad_alloc', BQP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BQP_inform%factorization_status,                &
                               mxGetPr( BQP_pointer%factorization_status ) )    
      CALL MATLAB_copy_to_ptr( BQP_inform%iter,                                &
                               mxGetPr( BQP_pointer%iter ) )             
      CALL MATLAB_copy_to_ptr( BQP_inform%cg_iter,                             &
                               mxGetPr( BQP_pointer%cg_iter ) )             
      CALL MATLAB_copy_to_ptr( BQP_inform%obj,                                 &
                               mxGetPr( BQP_pointer%obj ) )                     
      CALL MATLAB_copy_to_ptr( BQP_inform%norm_pg,                             &
                               mxGetPr( BQP_pointer%norm_pg ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( BQP_inform%time%total, wp ),              &
                               mxGetPr( BQP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( BQP_inform%time%analyse, wp ),            &
                               mxGetPr( BQP_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( BQP_inform%time%factorize, wp ),          &
                               mxGetPr( BQP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( BQP_inform%time%solve, wp ),              &
                               mxGetPr( BQP_pointer%time_pointer%solve ) )

!  positive-definite linear solvers

      CALL SBLS_matlab_inform_get( BQP_inform%SBLS_inform,                     &
                                   BQP_pointer%SBLS_pointer )

      RETURN

!  End of subroutine BQP_matlab_inform_get

      END SUBROUTINE BQP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_BQP_MATLAB_TYPES
