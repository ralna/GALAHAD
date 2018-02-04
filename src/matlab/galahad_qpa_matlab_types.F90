#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 04/03/2011 AT 10:15 GMT.

!-*-*-*-  G A L A H A D _ Q P A _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_QPA_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to QPA

      USE GALAHAD_MATLAB
      USE GALAHAD_SLS_MATLAB_TYPES
      USE GALAHAD_QPA_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPA_matlab_control_set, QPA_matlab_control_get,                &
                QPA_matlab_inform_create, QPA_matlab_inform_get

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

      TYPE, PUBLIC :: QPA_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, analyse, factorize, solve
        mwPointer :: clock_total, clock_preprocess
        mwPointer :: clock_analyse, clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: QPA_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: major_iter, iter, cg_iter, factorization_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: nfacts, nmods, num_g_infeas, num_b_infeas
        mwPointer :: obj, infeas_g, infeas_b, merit
        TYPE ( QPA_time_pointer_type ) :: time_pointer
        TYPE ( SLS_pointer_type ) :: SLS_pointer
      END TYPE 
    CONTAINS

!-*-  Q P A _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE QPA_matlab_control_set( ps, QPA_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QPA

!  Arguments

!  ps - given pointer to the structure
!  QPA_control - QPA control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( QPA_control_type ) :: QPA_control

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
                                 pc, QPA_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, QPA_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, QPA_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, QPA_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, QPA_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, QPA_control%maxit )
        CASE( 'factor' )
          CALL MATLAB_get_value( ps, 'factor',                                 &
                                 pc, QPA_control%factor )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, QPA_control%max_col )
        CASE( 'max_sc' )
          CALL MATLAB_get_value( ps, 'max_sc',                                 &
                                 pc, QPA_control%max_sc )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, QPA_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, QPA_control%valmin )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, QPA_control%itref_max )
        CASE( 'infeas_check_interval' )
          CALL MATLAB_get_value( ps, 'infeas_check_interval',                  &
                                 pc, QPA_control%infeas_check_interval )
        CASE( 'cg_maxit' )
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, QPA_control%cg_maxit )
        CASE( 'precon' )
          CALL MATLAB_get_value( ps, 'precon',                                 &
                                 pc, QPA_control%precon )
        CASE( 'nsemib' )
          CALL MATLAB_get_value( ps, 'nsemib',                                 &
                                 pc, QPA_control%nsemib )
        CASE( 'full_max_fill' )
          CALL MATLAB_get_value( ps, 'full_max_fill',                          &
                                 pc, QPA_control%full_max_fill )
        CASE( 'deletion_strategy' )
          CALL MATLAB_get_value( ps, 'deletion_strategy',                      &
                                 pc, QPA_control%deletion_strategy )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, QPA_control%restore_problem )
        CASE( 'monitor_residuals' )
          CALL MATLAB_get_value( ps, 'monitor_residuals',                      &
                                 pc, QPA_control%monitor_residuals )
        CASE( 'cold_start' )
          CALL MATLAB_get_value( ps, 'cold_start',                             &
                                 pc, QPA_control%cold_start )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, QPA_control%infinity )
        CASE( 'feas_tol' )
          CALL MATLAB_get_value( ps, 'feas_tol',                               &
                                 pc, QPA_control%feas_tol )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, QPA_control%obj_unbounded )
        CASE( 'increase_rho_g_factor' )
          CALL MATLAB_get_value( ps, 'increase_rho_g_factor',                  &
                                 pc, QPA_control%increase_rho_g_factor )
        CASE( 'infeas_g_improved_by_factor' )
          CALL MATLAB_get_value( ps,'infeas_g_improved_by_factor',             &
                                 pc, QPA_control%infeas_g_improved_by_factor )
        CASE( 'increase_rho_b_factor' )
          CALL MATLAB_get_value( ps, 'increase_rho_b_factor',                  &
                                 pc, QPA_control%increase_rho_b_factor )
        CASE( 'infeas_b_improved_by_factor' )
          CALL MATLAB_get_value( ps,'infeas_b_improved_by_factor',             &
                                 pc, QPA_control%infeas_b_improved_by_factor )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, QPA_control%pivot_tol )
        CASE( 'pivot_tol_for_dependencies' )
          CALL MATLAB_get_value( ps, 'pivot_tol_for_dependencies',             &
                                 pc, QPA_control%pivot_tol_for_dependencies )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, QPA_control%zero_pivot )
        CASE( 'inner_stop_relative' )
          CALL MATLAB_get_value( ps, 'inner_stop_relative',                    &
                                 pc, QPA_control%inner_stop_relative )
        CASE( 'inner_stop_absolute' )
          CALL MATLAB_get_value( ps, 'inner_stop_absolute',                    &
                                 pc, QPA_control%inner_stop_absolute )
        CASE( 'multiplier_tol' )
          CALL MATLAB_get_value( ps, 'multiplier_tol',                         &
                                 pc, QPA_control%multiplier_tol )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, QPA_control%treat_zero_bounds_as_general )
        CASE( 'solve_qp' )
          CALL MATLAB_get_value( ps, 'solve_qp',                               &
                                 pc, QPA_control%solve_qp )
        CASE( 'solve_within_bounds' )
          CALL MATLAB_get_value( ps, 'solve_within_bounds',                    &
                                 pc, QPA_control%solve_within_bounds )
        CASE( 'randomize' )
          CALL MATLAB_get_value( ps, 'randomize',                              &
                                 pc, QPA_control%randomize )
        CASE( 'array_syntax_worse_than_do_loop' )     
          CALL MATLAB_get_value( ps, 'array_syntax_worse_than_do_loop',        &
                                 pc,                                           &
                                 QPA_control%array_syntax_worse_than_do_loop )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, QPA_control%prefix, len )
        CASE( 'SLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SLS_control must be a structure' )
          CALL SLS_matlab_control_set( pc, QPA_control%SLS_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine QPA_matlab_control_set

      END SUBROUTINE QPA_matlab_control_set

!-*-  Q P A _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE QPA_matlab_control_get( struct, QPA_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QPA

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPA_control - QPA control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPA_control_type ) :: QPA_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 41
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'maxit                          ', &
         'factor                         ', 'max_col                        ', &
         'max_sc                         ', 'indmin                         ', &
         'valmin                         ', 'itref_max                      ', &
         'infeas_check_interval          ', 'cg_maxit                       ', &
         'precon                         ', 'nsemib                         ', &
         'full_max_fill                  ', 'deletion_strategy              ', &
         'restore_problem                ', 'monitor_residuals              ', &
         'cold_start                     ', 'infinity                       ', &
         'feas_tol                       ', 'obj_unbounded                  ', &
         'increase_rho_g_factor          ', 'infeas_g_improved_by_factor    ', &
         'increase_rho_b_factor          ', 'infeas_b_improved_by_factor    ', &
         'pivot_tol                      ', 'pivot_tol_for_dependencies     ', &
         'zero_pivot                     ', 'inner_stop_relative            ', &
         'inner_stop_absolute            ', 'multiplier_tol                 ', &
         'treat_zero_bounds_as_general   ', 'solve_qp                       ', &
         'solve_within_bounds            ', 'randomize                      ', &
         'array_syntax_worse_than_do_loop', 'prefix                         ', &
         'SLS_control                    ' /)

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
                                  QPA_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  QPA_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  QPA_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  QPA_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  QPA_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  QPA_control%maxit )
      CALL MATLAB_fill_component( pointer, 'factor',                           &
                                  QPA_control%factor )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  QPA_control%max_col )
      CALL MATLAB_fill_component( pointer, 'max_sc',                           &
                                  QPA_control%max_sc )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  QPA_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  QPA_control%valmin )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  QPA_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'infeas_check_interval',            &
                                  QPA_control%infeas_check_interval )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  QPA_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'precon',                           &
                                  QPA_control%precon )
      CALL MATLAB_fill_component( pointer, 'nsemib',                           &
                                  QPA_control%nsemib )
      CALL MATLAB_fill_component( pointer, 'full_max_fill',                    &
                                  QPA_control%full_max_fill )
      CALL MATLAB_fill_component( pointer, 'deletion_strategy',                &
                                  QPA_control%deletion_strategy )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  QPA_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'monitor_residuals',                &
                                  QPA_control%monitor_residuals )
      CALL MATLAB_fill_component( pointer, 'cold_start',                       &
                                  QPA_control%cold_start )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  QPA_control%infinity )
      CALL MATLAB_fill_component( pointer, 'feas_tol',                         &
                                  QPA_control%feas_tol )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  QPA_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'increase_rho_g_factor',            &
                                  QPA_control%increase_rho_g_factor )
      CALL MATLAB_fill_component( pointer, 'infeas_g_improved_by_factor',      &
                                  QPA_control%infeas_g_improved_by_factor )
      CALL MATLAB_fill_component( pointer, 'increase_rho_b_factor',            &
                                  QPA_control%increase_rho_b_factor )
      CALL MATLAB_fill_component( pointer, 'infeas_b_improved_by_factor',      &
                                  QPA_control%infeas_b_improved_by_factor )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  QPA_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_dependencies',       &
                                  QPA_control%pivot_tol_for_dependencies )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  QPA_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'inner_stop_relative',              &
                                  QPA_control%inner_stop_relative )
      CALL MATLAB_fill_component( pointer, 'inner_stop_absolute',              &
                                  QPA_control%inner_stop_absolute )
      CALL MATLAB_fill_component( pointer, 'multiplier_tol',                   &
                                  QPA_control%multiplier_tol )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  QPA_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'solve_qp',                         &
                                  QPA_control%solve_qp )
      CALL MATLAB_fill_component( pointer, 'solve_within_bounds',              &
                                  QPA_control%solve_within_bounds )
      CALL MATLAB_fill_component( pointer, 'randomize',                        &
                                  QPA_control%randomize )
      CALL MATLAB_fill_component( pointer, 'array_syntax_worse_than_do_loop',  &
                                  QPA_control%array_syntax_worse_than_do_loop )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  QPA_control%prefix )

!  create the components of sub-structure SLS_control

      CALL SLS_matlab_control_get( pointer, QPA_control%SLS_control,           &
                                   'SLS_control' )
      RETURN

!  End of subroutine QPA_matlab_control_get

      END SUBROUTINE QPA_matlab_control_get

!-*- Q P A _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE QPA_matlab_inform_create( struct, QPA_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold QPA_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPA_pointer - QPA pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPA_pointer_type ) :: QPA_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 19
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'major_iter           ', 'iter                 ',                   &
           'cg_iter              ', 'nfacts               ',                   &
           'nmods                ', 'factorization_status ',                   &
           'factorization_integer', 'factorization_real   ',                   &
           'num_g_infeas         ', 'num_b_infeas         ',                   &
           'obj                  ', 'infeas_g             ',                   &
           'infeas_b             ', 'merit                ',                   &
           'bad_alloc            ', 'time                 ',                   &
           'SLS_inform           '  /)
      INTEGER * 4, PARAMETER :: t_ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'analyse              ', 'factorize            ',                   &
           'solve                ', 'clock_total          ',                   &
           'clock_preprocess     ', 'clock_analyse        ',                   &
           'clock_factorize      ', 'clock_solve          '          /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, QPA_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        QPA_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'status', QPA_pointer%status )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
         'alloc_status', QPA_pointer%alloc_status )
      CALL MATLAB_create_char_component( QPA_pointer%pointer,                  &
        'bad_alloc', QPA_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
         'major_iter', QPA_pointer%major_iter )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
         'cg_iter', QPA_pointer%cg_iter )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'iter', QPA_pointer%iter )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'nfacts', QPA_pointer%nfacts )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'nmods', QPA_pointer%nmods )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'factorization_status', QPA_pointer%factorization_status )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'factorization_integer', QPA_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'factorization_real', QPA_pointer%factorization_real )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'num_g_infeas', QPA_pointer%num_g_infeas )
      CALL MATLAB_create_integer_component( QPA_pointer%pointer,               &
        'num_b_infeas', QPA_pointer%num_b_infeas )
      CALL MATLAB_create_real_component( QPA_pointer%pointer,                  &
        'obj', QPA_pointer%obj )
      CALL MATLAB_create_real_component( QPA_pointer%pointer,                  &
        'infeas_g', QPA_pointer%infeas_g )
      CALL MATLAB_create_real_component( QPA_pointer%pointer,                  &
        'infeas_b', QPA_pointer%infeas_b )
      CALL MATLAB_create_real_component( QPA_pointer%pointer,                  &
        'merit', QPA_pointer%merit )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( QPA_pointer%pointer,                    &
        'time', QPA_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'total', QPA_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'preprocess', QPA_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'analyse', QPA_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'factorize', QPA_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'solve', QPA_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'clock_total', QPA_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'clock_preprocess', QPA_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'clock_analyse', QPA_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'clock_factorize', QPA_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( QPA_pointer%time_pointer%pointer,     &
        'clock_solve', QPA_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SLS_inform

      CALL SLS_matlab_inform_create( QPA_pointer%pointer,                      &
                                     QPA_pointer%SLS_pointer, 'SLS_inform' )

      RETURN

!  End of subroutine QPA_matlab_inform_create

      END SUBROUTINE QPA_matlab_inform_create

!-*-*-  Q P A _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE QPA_matlab_inform_get( QPA_inform, QPA_pointer )

!  --------------------------------------------------------------

!  Set QPA_inform values from matlab pointers

!  Arguments

!  QPA_inform - QPA inform structure
!  QPA_pointer - QPA pointer structure

!  --------------------------------------------------------------

      TYPE ( QPA_inform_type ) :: QPA_inform
      TYPE ( QPA_pointer_type ) :: QPA_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( QPA_inform%status,                              &
                               mxGetPr( QPA_pointer%status ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%alloc_status,                        &
                               mxGetPr( QPA_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( QPA_pointer%pointer,                            &
                               'bad_alloc', QPA_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( QPA_inform%major_iter,                          &
                               mxGetPr( QPA_pointer%major_iter ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%iter,                                &
                               mxGetPr( QPA_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%cg_iter,                             &
                               mxGetPr( QPA_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%nfacts,                              &
                               mxGetPr( QPA_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%nmods,                               &
                               mxGetPr( QPA_pointer%nmods ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%factorization_status,                &
                              mxGetPr( QPA_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%factorization_integer,               &
                               mxGetPr( QPA_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%factorization_real,                  &
                               mxGetPr( QPA_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%num_g_infeas,                        &
                               mxGetPr( QPA_pointer%num_g_infeas ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%num_b_infeas,                        &
                               mxGetPr( QPA_pointer%num_b_infeas ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%obj,                                 &
                               mxGetPr( QPA_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%infeas_g,                            &
                               mxGetPr( QPA_pointer%infeas_g ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%infeas_b,                            &
                               mxGetPr( QPA_pointer%infeas_b ) )
      CALL MATLAB_copy_to_ptr( QPA_inform%merit,                               &
                               mxGetPr( QPA_pointer%merit ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%total, wp ),              &
                               mxGetPr( QPA_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%preprocess, wp ),         &
                               mxGetPr( QPA_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%analyse, wp ),            &
                               mxGetPr( QPA_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%factorize, wp ),          &
                               mxGetPr( QPA_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%solve, wp ),              &
                               mxGetPr( QPA_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%clock_total, wp ),        &
                         mxGetPr( QPA_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%clock_preprocess, wp ),   &
                         mxGetPr( QPA_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%clock_analyse, wp ),      &
                         mxGetPr( QPA_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%clock_factorize, wp ),    &
                         mxGetPr( QPA_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPA_inform%time%clock_solve, wp ),        &
                         mxGetPr( QPA_pointer%time_pointer%clock_solve ) )

!  indefinite linear solvers

      CALL SLS_matlab_inform_get( QPA_inform%SLS_inform,                       &
                                  QPA_pointer%SLS_pointer )

      RETURN

!  End of subroutine QPA_matlab_inform_get

      END SUBROUTINE QPA_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ Q P A _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_QPA_MATLAB_TYPES




