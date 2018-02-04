#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 04/03/2011 AT 10:30 GMT.

!-*-*-*-  G A L A H A D _ Q P C _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 15th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_QPC_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to QPC

      USE GALAHAD_MATLAB
      USE GALAHAD_QPA_MATLAB_TYPES
      USE GALAHAD_QPB_MATLAB_TYPES
      USE GALAHAD_CQP_MATLAB_TYPES
      USE GALAHAD_EQP_MATLAB_TYPES
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_QPC_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPC_matlab_control_set, QPC_matlab_control_get,                &
                QPC_matlab_inform_create, QPC_matlab_inform_get

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

      TYPE, PUBLIC :: QPC_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: QPC_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, factorization_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: nfacts, nmods, p_found, obj, non_negligible_pivot
        TYPE ( QPC_time_pointer_type ) :: time_pointer
        TYPE ( QPA_pointer_type ) :: QPA_pointer
        TYPE ( QPB_pointer_type ) :: QPB_pointer
        TYPE ( CQP_pointer_type ) :: CQP_pointer
        TYPE ( EQP_pointer_type ) :: EQP_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
      END TYPE 
    CONTAINS

!-*-  Q P C _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE QPC_matlab_control_set( ps, QPC_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QPC

!  Arguments

!  ps - given pointer to the structure
!  QPC_control - QPC control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( QPC_control_type ) :: QPC_control

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
                                 pc, QPC_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, QPC_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, QPC_control%print_level )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, QPC_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, QPC_control%valmin )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, QPC_control%restore_problem )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, QPC_control%infinity )
        CASE( 'identical_bounds_tol' )                                         
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, QPC_control%identical_bounds_tol )
        CASE( 'rho_g' )
          CALL MATLAB_get_value( ps, 'rho_g',                                  &
                                 pc, QPC_control%rho_g )
        CASE( 'rho_b' )                                                        
          CALL MATLAB_get_value( ps, 'rho_b',                                  &
                                 pc, QPC_control%rho_b )
        CASE( 'pivot_tol_for_dependencies' )                                   
          CALL MATLAB_get_value( ps, 'pivot_tol_for_dependencies',             &
                                 pc, QPC_control%pivot_tol_for_dependencies )
        CASE( 'zero_pivot' )                                                   
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, QPC_control%zero_pivot )
        CASE( 'cpu_time_limit' )                                               
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, QPC_control%cpu_time_limit )
        CASE( 'clock_time_limit' )         
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, QPC_control%clock_time_limit )
        CASE( 'treat_zero_bounds_as_general' )                                 
          CALL MATLAB_get_value( ps, 'treat_zero_bounds_as_general',           &
                                 pc, QPC_control%treat_zero_bounds_as_general ) 
        CASE( 'array_syntax_worse_than_do_loop' )                              
          CALL MATLAB_get_value( ps, 'array_syntax_worse_than_do_loop',        &
                                 pc,                                           &
                                 QPC_control%array_syntax_worse_than_do_loop )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, QPC_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, QPC_control%deallocate_error_fatal )
        CASE( 'no_qpa' )                                                       
          CALL MATLAB_get_value( ps, 'no_qpa',                                 &
                                 pc, QPC_control%no_qpa )
        CASE( 'no_qpb' )                                                       
          CALL MATLAB_get_value( ps, 'no_qpb',                                 &
                                 pc, QPC_control%no_qpb )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, QPC_control%prefix, len )
        CASE( 'QPA_control' )
          pc = mxGetField( ps, 1_mwi_, 'QPA_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component QPA_control must be a structure' )
          CALL QPA_matlab_control_set( pc, QPC_control%QPA_control, len )
        CASE( 'QPB_control' )
          pc = mxGetField( ps, 1_mwi_, 'QPB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component QPB_control must be a structure' )
          CALL QPB_matlab_control_set( pc, QPC_control%QPB_control, len )
        CASE( 'CQP_control' )
          pc = mxGetField( ps, 1_mwi_, 'CQP_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component CQP_control must be a structure' )
          CALL CQP_matlab_control_set( pc, QPC_control%CQP_control, len )
        CASE( 'EQP_control' )
          pc = mxGetField( ps, 1_mwi_, 'EQP_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component EQP_control must be a structure' )
          CALL EQP_matlab_control_set( pc, QPC_control%EQP_control, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, QPC_control%FDC_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine QPC_matlab_control_set

      END SUBROUTINE QPC_matlab_control_set

!-*-  Q P C _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE QPC_matlab_control_get( struct, QPC_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QPC

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPC_control - QPC control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPC_control_type ) :: QPC_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 26
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'indmin                         ', &
         'valmin                         ', 'restore_problem                ', &
         'infinity                       ', 'identical_bounds_tol           ', &
         'rho_g                          ', 'rho_b                          ', &
         'pivot_tol_for_dependencies     ', 'zero_pivot                     ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'treat_zero_bounds_as_general   ',                                    &
         'array_syntax_worse_than_do_loop', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'no_qpa                         ', &
         'no_qpb                         ', 'prefix                         ', &
         'QPA_control                    ', 'QPB_control                    ', &
         'CQP_control                    ', 'EQP_control                    ', &
         'FDC_control                    ' /)

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
                                  QPC_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  QPC_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  QPC_control%print_level )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  QPC_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  QPC_control%valmin )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  QPC_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  QPC_control%infinity )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  QPC_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'rho_g',                            &
                                  QPC_control%rho_g )
      CALL MATLAB_fill_component( pointer, 'rho_b',                            &
                                  QPC_control%rho_b )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_dependencies',       &
                                  QPC_control%pivot_tol_for_dependencies )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  QPC_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  QPC_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  QPC_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  QPC_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'array_syntax_worse_than_do_loop',  &
                                  QPC_control%array_syntax_worse_than_do_loop )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  QPC_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  QPC_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'no_qpa',                           &
                                  QPC_control%no_qpa )
      CALL MATLAB_fill_component( pointer, 'no_qpb',                           &
                                  QPC_control%no_qpb )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  QPC_control%prefix )

!  create the components of sub-structure QPA_control

      CALL QPA_matlab_control_get( pointer, QPC_control%QPA_control,           &
                                   'QPA_control' )

!  create the components of sub-structure QPB_control

      CALL QPB_matlab_control_get( pointer, QPC_control%QPB_control,           &
                                   'QPB_control' )

!  create the components of sub-structure CQP_control

      CALL CQP_matlab_control_get( pointer, QPC_control%CQP_control,           &
                                   'CQP_control' )

!  create the components of sub-structure EQP_control

      CALL EQP_matlab_control_get( pointer, QPC_control%EQP_control,           &
                                   'EQP_control' )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, QPC_control%FDC_control,           &
                                  'FDC_control' )

      RETURN

!  End of subroutine QPC_matlab_control_get

      END SUBROUTINE QPC_matlab_control_get

!-*- Q P C _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE QPC_matlab_inform_create( struct, QPC_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold QPC_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPC_pointer - QPC pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPC_pointer_type ) :: QPC_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 17
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_status ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'nfacts               ', 'nmods                ',                   &
           'p_found              ', 'obj                  ',                   &
           'non_negligible_pivot ', 'time                 ',                   &
           'QPA_inform           ', 'QPB_inform           ',                   &
           'CQP_inform           ', 'EQP_inform           ',                   &
           'FDC_inform           ' /) 
      INTEGER * 4, PARAMETER :: t_ninform = 12
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'find_dependent       ', 'analyse              ',                   &
           'factorize            ', 'solve                ',                   &
           'clock_total          ', 'clock_preprocess     ',                   &
           'clock_find_dependent ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, QPC_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        QPC_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'status', QPC_pointer%status )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
         'alloc_status', QPC_pointer%alloc_status )
      CALL MATLAB_create_char_component( QPC_pointer%pointer,                  &
        'bad_alloc', QPC_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'factorization_status', QPC_pointer%factorization_status )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'factorization_integer', QPC_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'factorization_real', QPC_pointer%factorization_real )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'nfacts', QPC_pointer%nfacts )
      CALL MATLAB_create_integer_component( QPC_pointer%pointer,               &
        'nmods', QPC_pointer%nmods )
      CALL MATLAB_create_logical_component( QPC_pointer%pointer,               &
        'p_found', QPC_pointer%p_found )
      CALL MATLAB_create_real_component( QPC_pointer%pointer,                  &
        'obj', QPC_pointer%obj )
      CALL MATLAB_create_real_component( QPC_pointer%pointer,                  &
         'non_negligible_pivot', QPC_pointer%non_negligible_pivot )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( QPC_pointer%pointer,                    &
        'time', QPC_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'total', QPC_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'preprocess', QPC_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'find_dependent', QPC_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'analyse', QPC_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'factorize', QPC_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'solve', QPC_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_total', QPC_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_preprocess', QPC_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_find_dependent', QPC_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_analyse', QPC_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_factorize', QPC_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( QPC_pointer%time_pointer%pointer,     &
        'clock_solve', QPC_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure QPA_inform

      CALL QPA_matlab_inform_create( QPC_pointer%pointer,                      &
                                     QPC_pointer%QPA_pointer, 'QPA_inform' )

!  Define the components of sub-structure QPB_inform

      CALL QPB_matlab_inform_create( QPC_pointer%pointer,                      &
                                     QPC_pointer%QPB_pointer, 'QPB_inform' )

!  Define the components of sub-structure CQP_inform

      CALL CQP_matlab_inform_create( QPC_pointer%pointer,                      &
                                     QPC_pointer%CQP_pointer, 'CQP_inform' )

!  Define the components of sub-structure EQP_inform

      CALL EQP_matlab_inform_create( QPC_pointer%pointer,                      &
                                     QPC_pointer%EQP_pointer, 'EQP_inform' )

!  Define the components of sub-structure FDC_inform 

      CALL FDC_matlab_inform_create( QPC_pointer%pointer,                      &
                                     QPC_pointer%FDC_pointer, 'FDC_inform' )

      RETURN

!  End of subroutine QPC_matlab_inform_create

      END SUBROUTINE QPC_matlab_inform_create

!-*-*-  Q P C _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE QPC_matlab_inform_get( QPC_inform, QPC_pointer )

!  --------------------------------------------------------------

!  Set QPC_inform values from matlab pointers

!  Arguments

!  QPC_inform - QPC inform structure
!  QPC_pointer - QPC pointer structure

!  --------------------------------------------------------------

      TYPE ( QPC_inform_type ) :: QPC_inform
      TYPE ( QPC_pointer_type ) :: QPC_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( QPC_inform%status,                              &
                               mxGetPr( QPC_pointer%status ) )
      CALL MATLAB_copy_to_ptr( QPC_inform%alloc_status,                        &
                               mxGetPr( QPC_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( QPC_pointer%pointer,                            &
                               'bad_alloc', QPC_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( QPC_inform%factorization_status,                &
                               mxGetPr( QPC_pointer%factorization_status ) )    
      CALL MATLAB_copy_to_ptr( QPC_inform%factorization_integer,               &
                               mxGetPr( QPC_pointer%factorization_integer ) )   
      CALL MATLAB_copy_to_ptr( QPC_inform%factorization_real,                  &
                               mxGetPr( QPC_pointer%factorization_real ) )      
      CALL MATLAB_copy_to_ptr( QPC_inform%nfacts,                              &
                               mxGetPr( QPC_pointer%nfacts ) )                  
      CALL MATLAB_copy_to_ptr( QPC_inform%nmods,                               &
                               mxGetPr( QPC_pointer%nmods ) )                   
      CALL MATLAB_copy_to_ptr( QPC_inform%p_found,                             &
                               mxGetPr( QPC_pointer%p_found ) )                
      CALL MATLAB_copy_to_ptr( QPC_inform%obj,                                 &
                               mxGetPr( QPC_pointer%obj ) )                     
      CALL MATLAB_copy_to_ptr( QPC_inform%non_negligible_pivot,                &
                               mxGetPr( QPC_pointer%non_negligible_pivot ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%total, wp ),              &
                               mxGetPr( QPC_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%preprocess, wp ),         &
                               mxGetPr( QPC_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%find_dependent, wp ),     &
                           mxGetPr( QPC_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%analyse, wp ),            &
                               mxGetPr( QPC_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%factorize, wp ),          &
                               mxGetPr( QPC_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%solve, wp ),              &
                               mxGetPr( QPC_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_total, wp ),        &
                      mxGetPr( QPC_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( QPC_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_find_dependent,wp), &
                      mxGetPr( QPC_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_analyse, wp ),      &
                      mxGetPr( QPC_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_factorize, wp ),    &
                      mxGetPr( QPC_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPC_inform%time%clock_solve, wp ),        &
                      mxGetPr( QPC_pointer%time_pointer%clock_solve ) )

!  active-set qp components

      CALL QPA_matlab_inform_get( QPC_inform%QPA_inform,                       &
                                  QPC_pointer%QPA_pointer )

!  interior-point qp components

      CALL QPB_matlab_inform_get( QPC_inform%QPB_inform,                       &
                                  QPC_pointer%QPB_pointer )

!  convex qp components

      CALL CQP_matlab_inform_get( QPC_inform%CQP_inform,                       &
                                  QPC_pointer%CQP_pointer )

!  equality qp components

      CALL EQP_matlab_inform_get( QPC_inform%EQP_inform,                       &
                                  QPC_pointer%EQP_pointer )

!  constraint-dependency check components

      CALL FDC_matlab_inform_get( QPC_inform%FDC_inform,                       &
                                  QPC_pointer%FDC_pointer )

      RETURN

!  End of subroutine QPC_matlab_inform_get

      END SUBROUTINE QPC_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ Q P C _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_QPC_MATLAB_TYPES
