#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 09:30 GMT.

!-*-*-*-  G A L A H A D _ W C P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 20th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_WCP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to WCP

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_WCP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: WCP_matlab_control_set, WCP_matlab_control_get,                &
                WCP_matlab_inform_create, WCP_matlab_inform_get

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

      TYPE, PUBLIC :: WCP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: WCP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, factorization_status
        mwPointer :: factorization_integer, factorization_real, nfacts
        mwPointer :: c_implicit, x_implicit, y_implicit, z_implicit
        mwPointer :: obj, non_negligible_pivot, feasible
        TYPE ( WCP_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
      END TYPE 
    CONTAINS

!-*-  W C P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE WCP_matlab_control_set( ps, WCP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to WCP

!  Arguments

!  ps - given pointer to the structure
!  WCP_control - WCP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( WCP_control_type ) :: WCP_control

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
                                 pc, WCP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, WCP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, WCP_control%print_level )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, WCP_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, WCP_control%valmin )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, WCP_control%restore_problem )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, WCP_control%start_print )
        CASE( 'stop_print ' )
          CALL MATLAB_get_value( ps, 'stop_print ',                            &
                                 pc, WCP_control%stop_print  )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, WCP_control%maxit )
        CASE( 'initial_point' )
          CALL MATLAB_get_value( ps, 'initial_point',                          &
                                 pc, WCP_control%initial_point )
        CASE( 'factor' )
          CALL MATLAB_get_value( ps, 'factor',                                 &
                                 pc, WCP_control%factor )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, WCP_control%max_col )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, WCP_control%itref_max )
        CASE( 'infeas_max' )
          CALL MATLAB_get_value( ps, 'infeas_max',                             &
                                 pc, WCP_control%infeas_max )
        CASE( 'perturbation_strategy' )
          CALL MATLAB_get_value( ps, 'perturbation_strategy',                  &
                                 pc, WCP_control%perturbation_strategy )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, WCP_control%infinity )
        CASE( 'identical_bounds_tol' )                                         
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, WCP_control%identical_bounds_tol )
        CASE( 'pivot_tol_for_dependencies' )                                   
          CALL MATLAB_get_value( ps, 'pivot_tol_for_dependencies',             &
                                 pc, WCP_control%pivot_tol_for_dependencies )
        CASE( 'zero_pivot' )                                                   
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, WCP_control%zero_pivot )
        CASE( 'cpu_time_limit' )                                               
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, WCP_control%cpu_time_limit )
        CASE( 'clock_time_limit' )         
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, WCP_control%clock_time_limit )
        CASE( 'stop_p' )
          CALL MATLAB_get_value( ps, 'stop_p',                                 &
                                 pc, WCP_control%stop_p )
        CASE( 'stop_d' )
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, WCP_control%stop_d )
        CASE( 'stop_c' )
          CALL MATLAB_get_value( ps, 'stop_c',                                 &
                                 pc, WCP_control%stop_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, WCP_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, WCP_control%dufeas )
        CASE( 'mu_target' )
          CALL MATLAB_get_value( ps, 'mu_target',                              &
                                 pc, WCP_control%mu_target )
        CASE( 'required_infeas_reduction' )
          CALL MATLAB_get_value( ps, 'required_infeas_reduction',              &
                                 pc, WCP_control%required_infeas_reduction )
        CASE( 'implicit_tol' )
          CALL MATLAB_get_value( ps, 'implicit_tol',                           &
                                 pc, WCP_control%implicit_tol )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, WCP_control%pivot_tol )
        CASE( 'perturb_start' )
          CALL MATLAB_get_value( ps, 'perturb_start',                          &
                                 pc, WCP_control%perturb_start )
        CASE( 'alpha_scale' )
          CALL MATLAB_get_value( ps, 'alpha_scale',                            &
                                 pc, WCP_control%alpha_scale )
        CASE( 'reduce_perturb_factor' )
          CALL MATLAB_get_value( ps, 'reduce_perturb_factor',                  &
                                 pc, WCP_control%reduce_perturb_factor )
        CASE( 'reduce_perturb_multiplier' )
          CALL MATLAB_get_value( ps, 'reduce_perturb_multiplier',              &
                                 pc, WCP_control%reduce_perturb_multiplier )
        CASE( 'mu_accept_fraction' )
          CALL MATLAB_get_value( ps, 'mu_accept_fraction',                     &
                                 pc, WCP_control%mu_accept_fraction )
        CASE( 'mu_increase_factor' )
          CALL MATLAB_get_value( ps, 'mu_increase_factor',                     &
                                 pc, WCP_control%mu_increase_factor )
        CASE( 'insufficiently_feasible' )
          CALL MATLAB_get_value( ps, 'insufficiently_feasible',                &
                                 pc, WCP_control%insufficiently_feasible )
        CASE( 'perturbation_small' )
          CALL MATLAB_get_value( ps, 'perturbation_small',                     &
                                 pc, WCP_control%perturbation_small )
        CASE( 'treat_zero_bounds_as_general' )                                 
          CALL MATLAB_get_value( ps, 'treat_zero_bounds_as_general',           &
                                 pc, WCP_control%treat_zero_bounds_as_general ) 
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, WCP_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, WCP_control%deallocate_error_fatal )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, WCP_control%remove_dependencies )
        CASE( 'just_feasible' )
          CALL MATLAB_get_value( ps, 'just_feasible',                          &
                                 pc, WCP_control%just_feasible )
        CASE( 'balance_initial_complementarity' )
          CALL MATLAB_get_value( ps, 'balance_initial_complementarity',        &
                                 pc,                                           &
                                 WCP_control%balance_initial_complementarity )
        CASE( 'use_corrector' )
          CALL MATLAB_get_value( ps, 'use_corrector',                          &
                                 pc, WCP_control%use_corrector )
        CASE( 'record_x_status' )
          CALL MATLAB_get_value( ps, 'record_x_status',                        &
                                 pc, WCP_control%record_x_status )
        CASE( 'record_c_status' )
          CALL MATLAB_get_value( ps, 'record_c_status',                        &
                                 pc, WCP_control%record_c_status )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, WCP_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, WCP_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, WCP_control%SBLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine WCP_matlab_control_set

      END SUBROUTINE WCP_matlab_control_set

!-*-  W C P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE WCP_matlab_control_get( struct, WCP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to WCP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  WCP_control - WCP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( WCP_control_type ) :: WCP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 50
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'indmin                         ', &
         'valmin                         ', 'restore_problem                ', &
         'start_print                    ', 'stop_print                     ', &
         'maxit                          ', 'initial_point                  ', &
         'factor                         ', 'max_col                        ', &
         'itref_max                      ', 'infeas_max                     ', &
         'perturbation_strategy          ', 'infinity                       ', &
         'identical_bounds_tol           ', 'pivot_tol_for_dependencies     ', &
         'zero_pivot                     ', 'cpu_time_limit                 ', &
         'clock_time_limit               ',                                    &
         'stop_p                         ', 'stop_d                         ', &
         'stop_c                         ', 'prfeas                         ', &
         'dufeas                         ', 'mu_target                      ', &
         'required_infeas_reduction      ', 'implicit_tol                   ', &
         'pivot_tol                      ', 'perturb_start                  ', &
         'alpha_scale                    ', 'reduce_perturb_factor          ', &
         'reduce_perturb_multiplier      ', 'mu_accept_fraction             ', &
         'mu_increase_factor             ', 'insufficiently_feasible        ', &
         'perturbation_small             ', 'treat_zero_bounds_as_general   ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'remove_dependencies            ', 'just_feasible                  ', &
         'balance_initial_complementarity', 'use_corrector                  ', &
         'record_x_status                ', 'record_c_status                ', &
         'prefix                         ', 'FDC_control                    ', &
         'SBLS_control                   ' /)

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
                                  WCP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  WCP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  WCP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  WCP_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  WCP_control%valmin )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  WCP_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  WCP_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print ',                      &
                                  WCP_control%stop_print  )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  WCP_control%maxit )
      CALL MATLAB_fill_component( pointer, 'initial_point',                    &
                                  WCP_control%initial_point )
      CALL MATLAB_fill_component( pointer, 'factor',                           &
                                  WCP_control%factor )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  WCP_control%max_col )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  WCP_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  WCP_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'perturbation_strategy',            &
                                  WCP_control%perturbation_strategy )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  WCP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  WCP_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_dependencies',       &
                                  WCP_control%pivot_tol_for_dependencies )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  WCP_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  WCP_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  WCP_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'stop_p',                           &
                                  WCP_control%stop_p )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  WCP_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_c',                           &
                                  WCP_control%stop_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  WCP_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  WCP_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'mu_target',                        &
                                  WCP_control%mu_target )
      CALL MATLAB_fill_component( pointer, 'required_infeas_reduction',        &
                                  WCP_control%required_infeas_reduction )
      CALL MATLAB_fill_component( pointer, 'implicit_tol',                     &
                                  WCP_control%implicit_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  WCP_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'perturb_start',                    &
                                  WCP_control%perturb_start )
      CALL MATLAB_fill_component( pointer, 'alpha_scale',                      &
                                  WCP_control%alpha_scale )
      CALL MATLAB_fill_component( pointer, 'reduce_perturb_factor',            &
                                  WCP_control%reduce_perturb_factor )
      CALL MATLAB_fill_component( pointer, 'reduce_perturb_multiplier',        &
                                  WCP_control%reduce_perturb_multiplier )
      CALL MATLAB_fill_component( pointer, 'mu_accept_fraction',               &
                                  WCP_control%mu_accept_fraction )
      CALL MATLAB_fill_component( pointer, 'mu_increase_factor',               &
                                  WCP_control%mu_increase_factor )
      CALL MATLAB_fill_component( pointer, 'insufficiently_feasible',          &
                                  WCP_control%insufficiently_feasible )
      CALL MATLAB_fill_component( pointer, 'perturbation_small',               &
                                  WCP_control%perturbation_small )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  WCP_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  WCP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  WCP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  WCP_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'just_feasible',                    &
                                  WCP_control%just_feasible )
      CALL MATLAB_fill_component( pointer, 'balance_initial_complementarity',  &
                                  WCP_control%balance_initial_complementarity )
      CALL MATLAB_fill_component( pointer, 'use_corrector',                    &
                                  WCP_control%use_corrector )
      CALL MATLAB_fill_component( pointer, 'record_x_status',                  &
                                  WCP_control%record_x_status )
      CALL MATLAB_fill_component( pointer, 'record_c_status',                  &
                                  WCP_control%record_c_status )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  WCP_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, WCP_control%FDC_control,           &
                                  'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, WCP_control%SBLS_control,         &
                                   'SBLS_control' )

      RETURN

!  End of subroutine WCP_matlab_control_get

      END SUBROUTINE WCP_matlab_control_get

!-*- W C P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE WCP_matlab_inform_create( struct, WCP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold WCP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  WCP_pointer - WCP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( WCP_pointer_type ) :: WCP_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 18
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'factorization_status ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'nfacts               ', 'iter                 ',                   &
           'obj                  ', 'non_negligible_pivot ',                   &
           'c_implicit           ', 'x_implicit           ',                   &
           'y_implicit           ', 'z_implicit           ',                   &
           'feasible             ', 'time                 ',                   &
           'FDC_inform           ', 'SBLS_inform          '      /)
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
        CALL MATLAB_create_substructure( struct, name, WCP_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        WCP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'status', WCP_pointer%status )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
         'alloc_status', WCP_pointer%alloc_status )
      CALL MATLAB_create_char_component( WCP_pointer%pointer,                  &
        'bad_alloc', WCP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'factorization_status', WCP_pointer%factorization_status )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'factorization_integer', WCP_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'factorization_real', WCP_pointer%factorization_real )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'iter', WCP_pointer%iter )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'nfacts', WCP_pointer%nfacts )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'c_implicit', WCP_pointer%c_implicit )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'x_implicit', WCP_pointer%x_implicit )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'y_implicit', WCP_pointer%y_implicit )
      CALL MATLAB_create_integer_component( WCP_pointer%pointer,               &
        'z_implicit', WCP_pointer%z_implicit )
      CALL MATLAB_create_logical_component( WCP_pointer%pointer,               &
        'feasible', WCP_pointer%feasible )
      CALL MATLAB_create_real_component( WCP_pointer%pointer,                  &
        'obj', WCP_pointer%obj )
      CALL MATLAB_create_real_component( WCP_pointer%pointer,                  &
         'non_negligible_pivot', WCP_pointer%non_negligible_pivot )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( WCP_pointer%pointer,                    &
        'time', WCP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'total', WCP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'preprocess', WCP_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'find_dependent', WCP_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'analyse', WCP_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'factorize', WCP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'solve', WCP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_total', WCP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_preprocess', WCP_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_find_dependent', WCP_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_analyse', WCP_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_factorize', WCP_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( WCP_pointer%time_pointer%pointer,     &
        'clock_solve', WCP_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( WCP_pointer%pointer,                      &
                                     WCP_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( WCP_pointer%pointer,                     &
                                      WCP_pointer%SBLS_pointer, 'SBLS_inform' )

      RETURN

!  End of subroutine WCP_matlab_inform_create

      END SUBROUTINE WCP_matlab_inform_create

!-*-*-  W C P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE WCP_matlab_inform_get( WCP_inform, WCP_pointer )

!  --------------------------------------------------------------

!  Set WCP_inform values from matlab pointers

!  Arguments

!  WCP_inform - WCP inform structure
!  WCP_pointer - WCP pointer structure

!  --------------------------------------------------------------

      TYPE ( WCP_inform_type ) :: WCP_inform
      TYPE ( WCP_pointer_type ) :: WCP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( WCP_inform%status,                              &
                               mxGetPr( WCP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%alloc_status,                        &
                               mxGetPr( WCP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( WCP_pointer%pointer,                            &
                               'bad_alloc', WCP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( WCP_inform%factorization_status,                &
                               mxGetPr( WCP_pointer%factorization_status ) )    
      CALL MATLAB_copy_to_ptr( WCP_inform%factorization_integer,               &
                               mxGetPr( WCP_pointer%factorization_integer ) )   
      CALL MATLAB_copy_to_ptr( WCP_inform%factorization_real,                  &
                               mxGetPr( WCP_pointer%factorization_real ) )      
      CALL MATLAB_copy_to_ptr( WCP_inform%iter,                                &
                               mxGetPr( WCP_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%nfacts,                              &
                               mxGetPr( WCP_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%c_implicit,                          &
                               mxGetPr( WCP_pointer%c_implicit ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%x_implicit,                          &
                               mxGetPr( WCP_pointer%x_implicit ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%y_implicit,                          &
                               mxGetPr( WCP_pointer%y_implicit ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%z_implicit,                          &
                               mxGetPr( WCP_pointer%z_implicit ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%feasible,                            &
                               mxGetPr( WCP_pointer%feasible ) )
      CALL MATLAB_copy_to_ptr( WCP_inform%obj,                                 &
                               mxGetPr( WCP_pointer%obj ) )                     
      CALL MATLAB_copy_to_ptr( WCP_inform%non_negligible_pivot,                &
                               mxGetPr( WCP_pointer%non_negligible_pivot ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%total, wp ),              &
                      mxGetPr( WCP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%preprocess, wp ),         &
                      mxGetPr( WCP_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%find_dependent, wp ),     &
                      mxGetPr( WCP_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%analyse, wp ),            &
                      mxGetPr( WCP_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%factorize, wp ),          &
                      mxGetPr( WCP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%solve, wp ),              &
                      mxGetPr( WCP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_total, wp ),        &
                      mxGetPr( WCP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( WCP_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_find_dependent,wp), &
                      mxGetPr( WCP_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_analyse, wp ),      &
                      mxGetPr( WCP_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_factorize, wp ),    &
                      mxGetPr( WCP_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( WCP_inform%time%clock_solve, wp ),        &
                      mxGetPr( WCP_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( WCP_inform%FDC_inform,                       &
                                  WCP_pointer%FDC_pointer )

!  preconditioner components

      CALL SBLS_matlab_inform_get( WCP_inform%SBLS_inform,                     &
                                   WCP_pointer%SBLS_pointer )

      RETURN

!  End of subroutine WCP_matlab_inform_get

      END SUBROUTINE WCP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ W C P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_WCP_MATLAB_TYPES


