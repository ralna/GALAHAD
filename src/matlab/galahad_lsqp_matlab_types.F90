#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 04/03/2011 AT 11:30 GMT.

!-*-*-*-  G A L A H A D _ L S Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LSQP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LSQP

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_LSQP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LSQP_matlab_control_set, LSQP_matlab_control_get,              &
                LSQP_matlab_inform_create, LSQP_matlab_inform_get

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

      TYPE, PUBLIC :: LSQP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, phase1_total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: LSQP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, factorization_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: nfacts, nbacts
        mwPointer :: obj, potential, non_negligible_pivot, feasible
        TYPE ( LSQP_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE 
    CONTAINS

!-*-  L S Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE LSQP_matlab_control_set( ps, LSQP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LSQP

!  Arguments

!  ps - given pointer to the structure
!  LSQP_control - LSQP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LSQP_control_type ) :: LSQP_control

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
                                 pc, LSQP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LSQP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LSQP_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, LSQP_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, LSQP_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, LSQP_control%maxit )
        CASE( 'factor' )
          CALL MATLAB_get_value( ps, 'factor',                                 &
                                 pc, LSQP_control%factor )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, LSQP_control%max_col )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, LSQP_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, LSQP_control%valmin )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, LSQP_control%itref_max )
        CASE( 'infeas_max' )
          CALL MATLAB_get_value( ps, 'infeas_max',                             &
                                 pc, LSQP_control%infeas_max )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                              pc, LSQP_control%indicator_type )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                              pc, LSQP_control%restore_problem)
        CASE( 'sif_file_device' )
          CALL MATLAB_get_value( ps, 'sif_file_device',                        &
                              pc, LSQP_control%sif_file_device )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, LSQP_control%infinity )
        CASE( 'stop_p' )
          CALL MATLAB_get_value( ps, 'stop_p',                                 &
                                 pc, LSQP_control%stop_p )
        CASE( 'stop_d' )
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, LSQP_control%stop_d )
        CASE( 'stop_c' )
          CALL MATLAB_get_value( ps, 'stop_c',                                 &
                                 pc, LSQP_control%stop_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, LSQP_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, LSQP_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, LSQP_control%muzero )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                              pc, LSQP_control%reduce_infeas )
        CASE( 'potential_unbounded' )
          CALL MATLAB_get_value( ps, 'potential_unbounded',                    &
                          pc, LSQP_control%potential_unbounded)
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, LSQP_control%pivot_tol )
        CASE( 'pivot_tol_for_dependencies' )
          CALL MATLAB_get_value( ps, 'pivot_tol_for_dependencies',             &
                                 pc, LSQP_control%pivot_tol_for_dependencies )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, LSQP_control%zero_pivot )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                        pc, LSQP_control%identical_bounds_tol )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                        pc, LSQP_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                        pc, LSQP_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                        pc, LSQP_control%indicator_tol_tapia )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                        pc, LSQP_control%cpu_time_limit )
        CASE( 'clock_time_limit' )         
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, LSQP_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                        pc, LSQP_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps, 'treat_zero_bounds_as_general',           &
             pc, LSQP_control%treat_zero_bounds_as_general )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                               pc, LSQP_control%just_feasible )
        CASE( 'getdua' )
          CALL MATLAB_get_value( ps, 'getdua',                                 &
                                 pc, LSQP_control%getdua )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, LSQP_control%feasol )
        CASE( 'balance_initial_complentarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complentarity',          &
                                 pc,                                           &
                                 LSQP_control%balance_initial_complentarity )
        CASE( 'use_corrector' )
          CALL MATLAB_get_value( ps, 'use_corrector',                          &
                              pc, LSQP_control%use_corrector )
        CASE( 'array_syntax_worse_than_do_loop' )     
          CALL MATLAB_get_value( ps, 'array_syntax_worse_than_do_loop',        &
                                 pc,                                           &
                                 LSQP_control%array_syntax_worse_than_do_loop )
        CASE( 'generate_sif_file' )
          CALL MATLAB_get_value( ps, 'generate_sif_file',                      &
                              pc, LSQP_control%generate_sif_file )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LSQP_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, LSQP_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, LSQP_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LSQP_matlab_control_set

      END SUBROUTINE LSQP_matlab_control_set

!-*-  L S Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LSQP_matlab_control_get( struct, LSQP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LSQP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSQP_control - LSQP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSQP_control_type ) :: LSQP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 45
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'maxit                          ', &
         'factor                         ', 'max_col                        ', &
         'indmin                         ', 'valmin                         ', &
         'itref_max                      ', 'infeas_max                     ', &
         'indicator_type                 ', 'restore_problem                ', &
         'sif_file_device                ', 'infinity                       ', &
         'stop_p                         ', 'stop_d                         ', &
         'stop_c                         ', 'prfeas                         ', &
         'dufeas                         ', 'muzero                         ', &
         'reduce_infeas                  ', 'potential_unbounded            ', &
         'pivot_tol                      ', 'pivot_tol_for_dependencies     ', &
         'zero_pivot                     ', 'identical_bounds_tol           ', &
         'indicator_tol_p                ', 'indicator_tol_pd               ', &
         'indicator_tol_tapia            ', 'cpu_time_limit                 ', &
         'clock_time_limit               ',                                    &
         'remove_dependencies            ', 'treat_zero_bounds_as_general   ', &
         'just_feasible                  ', 'getdua                         ', &
         'feasol                         ', 'balance_initial_complentarity  ', &
         'use_corrector                  ', 'array_syntax_worse_than_do_loop', &
         'generate_sif_file              ', 'prefix                         ', &
         'FDC_control                    ', 'SBLS_control                   ' /)

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
                                  LSQP_control%error )
      CALL MATLAB_fill_component( pointer, 'error',                            &
                                  LSQP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LSQP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LSQP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  LSQP_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  LSQP_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  LSQP_control%maxit )
      CALL MATLAB_fill_component( pointer, 'factor',                           &
                                  LSQP_control%factor )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  LSQP_control%max_col )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  LSQP_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  LSQP_control%valmin )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  LSQP_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  LSQP_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  LSQP_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  LSQP_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'sif_file_device',                  &
                                  LSQP_control%sif_file_device )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  LSQP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_p',                           &
                                  LSQP_control%stop_p )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  LSQP_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_c',                           &
                                  LSQP_control%stop_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  LSQP_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  LSQP_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  LSQP_control%muzero )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  LSQP_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'potential_unbounded',              &
                                  LSQP_control%potential_unbounded )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  LSQP_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_dependencies',       &
                                  LSQP_control%pivot_tol_for_dependencies )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  LSQP_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  LSQP_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  LSQP_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  LSQP_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  LSQP_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  LSQP_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  LSQP_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  LSQP_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  LSQP_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  LSQP_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'getdua',                           &
                                  LSQP_control%getdua )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  LSQP_control%feasol )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complentarity',    &
                                  LSQP_control%balance_initial_complentarity )
      CALL MATLAB_fill_component( pointer, 'use_corrector',                    &
                                  LSQP_control%use_corrector )
      CALL MATLAB_fill_component( pointer, 'array_syntax_worse_than_do_loop',  &
                                  LSQP_control%array_syntax_worse_than_do_loop )
      CALL MATLAB_fill_component( pointer, 'generate_sif_file',                &
                                  LSQP_control%generate_sif_file )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LSQP_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, LSQP_control%FDC_control,          &
                                  'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, LSQP_control%SBLS_control,        &
                                    'SBLS_control' )

      RETURN

!  End of subroutine LSQP_matlab_control_get

      END SUBROUTINE LSQP_matlab_control_get

!-*- L S Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE LSQP_matlab_inform_create( struct, LSQP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LSQP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSQP_pointer - LSQP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSQP_pointer_type ) :: LSQP_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'iter                 ', 'factorization_status ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'nfacts               ', 'nbacts               ',                   &
           'obj                  ', 'potential            ',                   &
           'non_negligible_pivot ', 'feasible             ',                   &
           'bad_alloc            ', 'time                 ',                   &
           'FDC_inform           ', 'SBLS_inform          '     /)
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
        CALL MATLAB_create_substructure( struct, name, LSQP_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LSQP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
        'status', LSQP_pointer%status )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'alloc_status', LSQP_pointer%alloc_status )
      CALL MATLAB_create_char_component( LSQP_pointer%pointer,                 &
        'bad_alloc', LSQP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'iter', LSQP_pointer%iter )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'factorization_status', LSQP_pointer%factorization_status )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'factorization_integer', LSQP_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'factorization_real', LSQP_pointer%factorization_real )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'nfacts', LSQP_pointer%nfacts )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'nbacts', LSQP_pointer%nbacts )
      CALL MATLAB_create_real_component( LSQP_pointer%pointer,                 &
         'obj', LSQP_pointer%obj )
      CALL MATLAB_create_integer_component( LSQP_pointer%pointer,              &
         'potential', LSQP_pointer%potential )
      CALL MATLAB_create_real_component( LSQP_pointer%pointer,                 &
          'non_negligible_pivot', LSQP_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( LSQP_pointer%pointer,              &
         'feasible', LSQP_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( LSQP_pointer%pointer,                   &
        'time', LSQP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'total', LSQP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'preprocess', LSQP_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'find_dependent', LSQP_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'analyse', LSQP_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'factorize', LSQP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'solve', LSQP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_total', LSQP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_preprocess', LSQP_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_find_dependent', LSQP_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_analyse', LSQP_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_factorize', LSQP_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( LSQP_pointer%time_pointer%pointer,    &
        'clock_solve', LSQP_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( LSQP_pointer%pointer,                     &
                                     LSQP_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( LSQP_pointer%pointer,                    &
                                      LSQP_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine LSQP_matlab_inform_create

      END SUBROUTINE LSQP_matlab_inform_create

!-*-*-  L S Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE LSQP_matlab_inform_get( LSQP_inform, LSQP_pointer )

!  --------------------------------------------------------------

!  Set LSQP_inform values from matlab pointers

!  Arguments

!  LSQP_inform - LSQP inform structure
!  LSQP_pointer - LSQP pointer structure

!  --------------------------------------------------------------

      TYPE ( LSQP_inform_type ) :: LSQP_inform
      TYPE ( LSQP_pointer_type ) :: LSQP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LSQP_inform%status,                             &
                               mxGetPr( LSQP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LSQP_inform%alloc_status,                       &
                               mxGetPr( LSQP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LSQP_pointer%pointer,                           &
                               'bad_alloc', LSQP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LSQP_inform%iter,                               &
                               mxGetPr( LSQP_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( LSQP_inform%factorization_status,               &
                               mxGetPr( LSQP_pointer%factorization_status ) )   
      CALL MATLAB_copy_to_ptr( LSQP_inform%factorization_integer,              &
                               mxGetPr( LSQP_pointer%factorization_integer ) )  
      CALL MATLAB_copy_to_ptr( LSQP_inform%factorization_real,                 &
                               mxGetPr( LSQP_pointer%factorization_real ) )     
      CALL MATLAB_copy_to_ptr( LSQP_inform%nfacts,                             &
                               mxGetPr( LSQP_pointer%nfacts ) )                 
      CALL MATLAB_copy_to_ptr( LSQP_inform%nbacts,                             &
                               mxGetPr( LSQP_pointer%nbacts ) )                 
      CALL MATLAB_copy_to_ptr( LSQP_inform%obj,                                &
                               mxGetPr( LSQP_pointer%obj ) )                    
      CALL MATLAB_copy_to_ptr( LSQP_inform%potential,                          &
                               mxGetPr( LSQP_pointer%potential ) )         
      CALL MATLAB_copy_to_ptr( LSQP_inform%non_negligible_pivot,               &
                               mxGetPr( LSQP_pointer%non_negligible_pivot ) ) 
      CALL MATLAB_copy_to_ptr( LSQP_inform%feasible,                           &
                               mxGetPr( LSQP_pointer%feasible ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%total, wp ),             &
                       mxGetPr( LSQP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%preprocess, wp ),        &
                       mxGetPr( LSQP_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%find_dependent, wp ),    &
                      mxGetPr( LSQP_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%analyse, wp ),           &
                      mxGetPr( LSQP_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%factorize, wp ),         &
                      mxGetPr( LSQP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%solve, wp ),             &
                      mxGetPr( LSQP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_total, wp ),       &
                      mxGetPr( LSQP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_preprocess, wp ),  &
                      mxGetPr( LSQP_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_find_dependent,wp),&
                      mxGetPr( LSQP_pointer%time_pointer%clock_find_dependent ))
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_analyse, wp ),     &
                      mxGetPr( LSQP_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_factorize, wp ),   &
                      mxGetPr( LSQP_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( LSQP_inform%time%clock_solve, wp ),       &
                      mxGetPr( LSQP_pointer%time_pointer%clock_solve ) )

!  constraint-dependency check components

      CALL FDC_matlab_inform_get( LSQP_inform%FDC_inform,                      &
                                  LSQP_pointer%FDC_pointer )

!  linear system solver components

       CALL SBLS_matlab_inform_get( LSQP_inform%SBLS_inform,                   &
                                    LSQP_pointer%SBLS_pointer )

      RETURN

!  End of subroutine LSQP_matlab_inform_get

      END SUBROUTINE LSQP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L S Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LSQP_MATLAB_TYPES
