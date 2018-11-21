#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 03/03/2011 AT 13:30 GMT.

!-*-*-*-  G A L A H A D _ Q P B _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_QPB_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to QPB

      USE GALAHAD_MATLAB
      USE GALAHAD_LSQP_MATLAB_TYPES
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_QPB_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPB_matlab_control_set, QPB_matlab_control_get,                &
                QPB_matlab_inform_create, QPB_matlab_inform_get

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

      TYPE, PUBLIC :: QPB_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess, find_dependent
        mwPointer :: analyse, factorize, solve
        mwPointer :: phase1_total, phase1_analyse
        mwPointer :: phase1_factorize, phase1_solve
        mwPointer :: clock_total, clock_preprocess, clock_find_dependent
        mwPointer :: clock_analyse, clock_factorize, clock_solve
        mwPointer :: clock_phase1_total, clock_phase1_analyse
        mwPointer :: clock_phase1_factorize, clock_phase1_solve
      END TYPE

      TYPE, PUBLIC :: QPB_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, cg_iter, factorization_status
        mwPointer :: factorization_integer, factorization_real
        mwPointer :: nfacts, nbacts, nmods
        mwPointer :: obj, non_negligible_pivot, feasible
        TYPE ( QPB_time_pointer_type ) :: time_pointer
        TYPE ( LSQP_pointer_type ) :: LSQP_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
        TYPE ( GLTR_pointer_type ) :: GLTR_pointer
      END TYPE
    CONTAINS

!-*-  Q P B _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE QPB_matlab_control_set( ps, QPB_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QPB

!  Arguments

!  ps - given pointer to the structure
!  QPB_control - QPB control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( QPB_control_type ) :: QPB_control

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
                                 pc, QPB_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, QPB_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, QPB_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, QPB_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, QPB_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, QPB_control%maxit  )
        CASE( 'factor' )
          CALL MATLAB_get_value( ps, 'factor',                                 &
                                 pc, QPB_control%factor )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, QPB_control%max_col )
        CASE( 'indmin' )
          CALL MATLAB_get_value( ps, 'indmin',                                 &
                                 pc, QPB_control%indmin )
        CASE( 'valmin' )
          CALL MATLAB_get_value( ps, 'valmin',                                 &
                                 pc, QPB_control%valmin )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, QPB_control%itref_max )
        CASE( 'infeas_max ' )
          CALL MATLAB_get_value( ps, 'infeas_max ',                            &
                                 pc, QPB_control%infeas_max  )
        CASE( 'cg_maxit' )
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, QPB_control%cg_maxit )
        CASE( 'precon' )
          CALL MATLAB_get_value( ps, 'precon',                                 &
                                 pc, QPB_control%precon )
        CASE( 'nsemib' )
          CALL MATLAB_get_value( ps, 'nsemib',                                 &
                                 pc, QPB_control%nsemib )
        CASE( 'indicator_type' )
          CALL MATLAB_get_value( ps, 'indicator_type',                         &
                                 pc, QPB_control%indicator_type )
        CASE( 'extrapolate' )
          CALL MATLAB_get_value( ps, 'extrapolate',                            &
                                 pc, QPB_control%extrapolate )
        CASE( 'path_history' )
          CALL MATLAB_get_value( ps, 'path_history',                           &
                                 pc, QPB_control%path_history )
        CASE( 'path_derivatives' )
          CALL MATLAB_get_value( ps, 'path_derivatives',                       &
                                 pc, QPB_control%path_derivatives )
        CASE( 'fit_order' )
          CALL MATLAB_get_value( ps, 'fit_order',                              &
                                 pc, QPB_control%fit_order )
        CASE( 'restore_problem' )
          CALL MATLAB_get_value( ps, 'restore_problem',                        &
                                 pc, QPB_control%restore_problem )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, QPB_control%infinity )
        CASE( 'stop_p' )
          CALL MATLAB_get_value( ps, 'stop_p',                                 &
                                 pc, QPB_control%stop_p )
        CASE( 'stop_d' )
          CALL MATLAB_get_value( ps, 'stop_d',                                 &
                                 pc, QPB_control%stop_d )
        CASE( 'stop_c' )
          CALL MATLAB_get_value( ps, 'stop_c',                                 &
                                 pc, QPB_control%stop_c )
        CASE( 'prfeas' )
          CALL MATLAB_get_value( ps, 'prfeas',                                 &
                                 pc, QPB_control%prfeas )
        CASE( 'dufeas' )
          CALL MATLAB_get_value( ps, 'dufeas',                                 &
                                 pc, QPB_control%dufeas )
        CASE( 'muzero' )
          CALL MATLAB_get_value( ps, 'muzero',                                 &
                                 pc, QPB_control%muzero )
        CASE( 'reduce_infeas' )
          CALL MATLAB_get_value( ps, 'reduce_infeas',                          &
                                 pc, QPB_control%reduce_infeas )
        CASE( 'obj_unbounded' )
          CALL MATLAB_get_value( ps, 'obj_unbounded',                          &
                                 pc, QPB_control%obj_unbounded )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, QPB_control%pivot_tol )
        CASE( 'pivot_tol_for_dependencies' )
          CALL MATLAB_get_value( ps, 'pivot_tol_for_dependencies',             &
                                 pc, QPB_control%pivot_tol_for_dependencies )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, QPB_control%zero_pivot )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, QPB_control%identical_bounds_tol )
        CASE( 'indicator_tol_p' )
          CALL MATLAB_get_value( ps, 'indicator_tol_p',                        &
                                 pc, QPB_control%indicator_tol_p )
        CASE( 'indicator_tol_pd' )
          CALL MATLAB_get_value( ps, 'indicator_tol_pd',                       &
                                 pc, QPB_control%indicator_tol_pd )
        CASE( 'indicator_tol_tapia' )
          CALL MATLAB_get_value( ps, 'indicator_tol_tapia',                    &
                                 pc, QPB_control%indicator_tol_tapia )
        CASE( 'inner_stop_relative' )
          CALL MATLAB_get_value( ps, 'inner_stop_relative',                    &
                                 pc, QPB_control%inner_stop_relative )
        CASE( 'inner_stop_absolute' )
          CALL MATLAB_get_value( ps, 'inner_stop_absolute',                    &
                                 pc, QPB_control%inner_stop_absolute )
        CASE( 'initial_radius' )
          CALL MATLAB_get_value( ps, 'initial_radius',                         &
                                 pc, QPB_control%initial_radius )
        CASE( 'inner_fraction_opt' )
          CALL MATLAB_get_value( ps, 'inner_fraction_opt',                     &
                                 pc, QPB_control%inner_fraction_opt )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, QPB_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, QPB_control%clock_time_limit )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, QPB_control%remove_dependencies )
        CASE( 'treat_zero_bounds_as_general' )
          CALL MATLAB_get_value( ps,'treat_zero_bounds_as_general',            &
                                 pc, QPB_control%treat_zero_bounds_as_general )
        CASE( 'center' )
          CALL MATLAB_get_value( ps, 'center',                                 &
                                 pc, QPB_control%center )
        CASE( 'primal' )
          CALL MATLAB_get_value( ps, 'primal',                                 &
                                 pc, QPB_control%primal )
        CASE( 'feasol' )
          CALL MATLAB_get_value( ps, 'feasol',                                 &
                                 pc, QPB_control%feasol )
        CASE( 'array_syntax_worse_than_do_loop' )
          CALL MATLAB_get_value( ps, 'array_syntax_worse_than_do_loop',        &
                                 pc,                                           &
                                 QPB_control%array_syntax_worse_than_do_loop )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, QPB_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, QPB_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, QPB_control%prefix, len )
        CASE( 'LSQP_control' )
          pc = mxGetField( ps, 1_mwi_, 'LSQP_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component LSQP_control must be a structure' )
          CALL LSQP_matlab_control_set( pc, QPB_control%LSQP_control, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, QPB_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, QPB_control%SBLS_control, len )
        CASE( 'GLTR_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLTR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLTR_control must be a structure' )
          CALL GLTR_matlab_control_set( pc, QPB_control%GLTR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine QPB_matlab_control_set

      END SUBROUTINE QPB_matlab_control_set

!-*-  Q P B _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE QPB_matlab_control_get( struct, QPB_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QPB

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPB_control - QPB control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPB_control_type ) :: QPB_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 56
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'maxit                          ', &
         'factor                         ', 'max_col                        ', &
         'indmin                         ', 'valmin                         ', &
         'itref_max                      ', 'infeas_max                     ', &
         'cg_maxit                       ', 'precon                         ', &
         'nsemib                         ', 'indicator_type                 ', &
         'extrapolate                    ', 'path_history                   ', &
         'path_derivatives               ', 'fit_order                      ', &
         'restore_problem                ', 'infinity                       ', &
         'stop_p                         ', 'stop_d                         ', &
         'stop_c                         ', 'prfeas                         ', &
         'dufeas                         ', 'muzero                         ', &
         'reduce_infeas                  ', 'obj_unbounded                  ', &
         'pivot_tol                      ', 'pivot_tol_for_dependencies     ', &
         'zero_pivot                     ', 'identical_bounds_tol           ', &
         'indicator_tol_p                ', 'indicator_tol_pd               ', &
         'indicator_tol_tapia            ', 'inner_stop_relative            ', &
         'inner_stop_absolute            ', 'initial_radius                 ', &
         'inner_fraction_opt             ', 'cpu_time_limit                 ', &
         'clock_time_limit               ',                                    &
         'remove_dependencies            ', 'treat_zero_bounds_as_general   ', &
         'center                         ', 'primal                         ', &
         'feasol                         ', 'array_syntax_worse_than_do_loop', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         ', 'LSQP_control                   ', &
         'FDC_control                    ', 'SBLS_control                   ', &
         'GLTR_control                   '  /)

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
                                  QPB_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  QPB_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  QPB_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  QPB_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  QPB_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  QPB_control%maxit )
      CALL MATLAB_fill_component( pointer, 'factor',                           &
                                  QPB_control%factor )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  QPB_control%max_col )
      CALL MATLAB_fill_component( pointer, 'indmin',                           &
                                  QPB_control%indmin )
      CALL MATLAB_fill_component( pointer, 'valmin',                           &
                                  QPB_control%valmin )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  QPB_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'infeas_max',                       &
                                  QPB_control%infeas_max )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  QPB_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'precon',                           &
                                  QPB_control%precon )
      CALL MATLAB_fill_component( pointer, 'nsemib',                           &
                                  QPB_control%nsemib )
      CALL MATLAB_fill_component( pointer, 'indicator_type',                   &
                                  QPB_control%indicator_type )
      CALL MATLAB_fill_component( pointer, 'extrapolate',                      &
                                  QPB_control%extrapolate )
      CALL MATLAB_fill_component( pointer, 'path_history',                     &
                                  QPB_control%path_history )
      CALL MATLAB_fill_component( pointer, 'path_derivatives',                 &
                                  QPB_control%path_derivatives )
      CALL MATLAB_fill_component( pointer, 'fit_order',                        &
                                  QPB_control%fit_order )
      CALL MATLAB_fill_component( pointer, 'restore_problem',                  &
                                  QPB_control%restore_problem )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  QPB_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_p',                           &
                                  QPB_control%stop_p )
      CALL MATLAB_fill_component( pointer, 'stop_d',                           &
                                  QPB_control%stop_d )
      CALL MATLAB_fill_component( pointer, 'stop_c',                           &
                                  QPB_control%stop_c )
      CALL MATLAB_fill_component( pointer, 'prfeas',                           &
                                  QPB_control%prfeas )
      CALL MATLAB_fill_component( pointer, 'dufeas',                           &
                                  QPB_control%dufeas )
      CALL MATLAB_fill_component( pointer, 'muzero',                           &
                                  QPB_control%muzero )
      CALL MATLAB_fill_component( pointer, 'reduce_infeas',                    &
                                  QPB_control%reduce_infeas )
      CALL MATLAB_fill_component( pointer, 'obj_unbounded',                    &
                                  QPB_control%obj_unbounded )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  QPB_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_dependencies',       &
                                  QPB_control%pivot_tol_for_dependencies )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  QPB_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  QPB_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_p',                  &
                                  QPB_control%indicator_tol_p )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_pd',                 &
                                  QPB_control%indicator_tol_pd )
      CALL MATLAB_fill_component( pointer, 'indicator_tol_tapia',              &
                                  QPB_control%indicator_tol_tapia )
      CALL MATLAB_fill_component( pointer, 'inner_stop_relative',              &
                                  QPB_control%inner_stop_relative )
      CALL MATLAB_fill_component( pointer, 'inner_stop_absolute',              &
                                  QPB_control%inner_stop_absolute )
      CALL MATLAB_fill_component( pointer, 'initial_radius',                   &
                                  QPB_control%initial_radius )
      CALL MATLAB_fill_component( pointer, 'inner_fraction_opt',               &
                                  QPB_control%inner_fraction_opt )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  QPB_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  QPB_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  QPB_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'treat_zero_bounds_as_general',     &
                                  QPB_control%treat_zero_bounds_as_general )
      CALL MATLAB_fill_component( pointer, 'center',                           &
                                  QPB_control%center )
      CALL MATLAB_fill_component( pointer, 'primal',                           &
                                  QPB_control%primal )
      CALL MATLAB_fill_component( pointer, 'feasol',                           &
                                  QPB_control%feasol )
      CALL MATLAB_fill_component( pointer, 'array_syntax_worse_than_do_loop',  &
                                  QPB_control%array_syntax_worse_than_do_loop )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  QPB_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  QPB_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  QPB_control%prefix )

!  create the components of sub-structure LSQP_control

      CALL LSQP_matlab_control_get( pointer, QPB_control%LSQP_control,         &
                                    'LSQP_control' )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, QPB_control%FDC_control,           &
                                  'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, QPB_control%SBLS_control,         &
                                    'SBLS_control' )

!  create the components of sub-structure GLTR_control

      CALL GLTR_matlab_control_get( pointer, QPB_control%GLTR_control,         &
                                    'GLTR_control' )

      RETURN

!  End of subroutine QPB_matlab_control_get

      END SUBROUTINE QPB_matlab_control_get

!-*- Q P B _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE QPB_matlab_inform_create( struct, QPB_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold QPB_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QPB_pointer - QPB pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QPB_pointer_type ) :: QPB_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 19
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'iter                 ', 'cg_iter              ',                   &
           'factorization_status ', 'factorization_integer',                   &
           'factorization_real   ', 'nfacts               ',                   &
           'nbacts               ', 'nmods                ',                   &
           'obj                  ', 'non_negligible_pivot ',                   &
           'feasible             ', 'bad_alloc            ',                   &
           'time                 ', 'LSQP_inform          ',                   &
           'FDC_inform           ', 'SBLS_inform          ',                   &
           'GLTR_inform          '      /)
      INTEGER * 4, PARAMETER :: t_ninform = 20
      CHARACTER ( LEN = 22 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                 ', 'preprocess            ',                 &
           'find_dependent        ', 'analyse               ',                 &
           'factorize             ', 'solve                 ',                 &
           'phase1_total          ', 'phase1_analyse        ',                 &
           'phase1_factorize      ', 'phase1_solve          ',                 &
           'clock_total           ', 'clock_preprocess      ',                 &
           'clock_find_dependent  ', 'clock_analyse         ',                 &
           'clock_factorize       ', 'clock_solve           ',                 &
           'clock_phase1_total    ', 'clock_phase1_analyse  ',                 &
           'clock_phase1_factorize', 'clock_phase1_solve    '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, QPB_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        QPB_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
        'status', QPB_pointer%status )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'alloc_status', QPB_pointer%alloc_status )
      CALL MATLAB_create_char_component( QPB_pointer%pointer,                  &
        'bad_alloc', QPB_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'iter', QPB_pointer%iter )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'cg_iter', QPB_pointer%cg_iter )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'factorization_status', QPB_pointer%factorization_status )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'factorization_integer', QPB_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'factorization_real', QPB_pointer%factorization_real )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'nfacts', QPB_pointer%nfacts )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'nbacts', QPB_pointer%nbacts )
      CALL MATLAB_create_integer_component( QPB_pointer%pointer,               &
         'nmods', QPB_pointer%nmods )
      CALL MATLAB_create_real_component( QPB_pointer%pointer,                  &
         'obj', QPB_pointer%obj )
      CALL MATLAB_create_real_component( QPB_pointer%pointer,                  &
          'non_negligible_pivot', QPB_pointer%non_negligible_pivot )
      CALL MATLAB_create_logical_component( QPB_pointer%pointer,               &
         'feasible', QPB_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( QPB_pointer%pointer,                    &
        'time', QPB_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'total', QPB_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'preprocess', QPB_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'find_dependent', QPB_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'analyse', QPB_pointer%time_pointer%analyse )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'factorize', QPB_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'solve', QPB_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'phase1_total', QPB_pointer%time_pointer%phase1_total )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'phase1_analyse', QPB_pointer%time_pointer%phase1_analyse )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'phase1_factorize', QPB_pointer%time_pointer%phase1_factorize )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'phase1_solve', QPB_pointer%time_pointer%phase1_solve )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_total', QPB_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_preprocess', QPB_pointer%time_pointer%clock_preprocess )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_find_dependent', QPB_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_analyse', QPB_pointer%time_pointer%clock_analyse )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_factorize', QPB_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_solve', QPB_pointer%time_pointer%clock_solve )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_phase1_total', QPB_pointer%time_pointer%clock_phase1_total )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_phase1_analyse', QPB_pointer%time_pointer%clock_phase1_analyse )
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
      'clock_phase1_factorize', QPB_pointer%time_pointer%clock_phase1_factorize)
      CALL MATLAB_create_real_component( QPB_pointer%time_pointer%pointer,     &
        'clock_phase1_solve', QPB_pointer%time_pointer%clock_phase1_solve )

!  Define the components of sub-structure LSQP_inform

      CALL LSQP_matlab_inform_create( QPB_pointer%pointer,                     &
                                      QPB_pointer%LSQP_pointer, 'LSQP_inform' )

!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( QPB_pointer%pointer,                      &
                                     QPB_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( QPB_pointer%pointer,                     &
                                      QPB_pointer%SBLS_pointer, 'SBLS_inform' )

!  Define the components of sub-structure GLTR_inform

      CALL GLTR_matlab_inform_create( QPB_pointer%pointer,                     &
                                      QPB_pointer%GLTR_pointer, 'GLTR_inform' )

      RETURN

!  End of subroutine QPB_matlab_inform_create

      END SUBROUTINE QPB_matlab_inform_create

!-*-*-  Q P B _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE QPB_matlab_inform_get( QPB_inform, QPB_pointer )

!  --------------------------------------------------------------

!  Set QPB_inform values from matlab pointers

!  Arguments

!  QPB_inform - QPB inform structure
!  QPB_pointer - QPB pointer structure

!  --------------------------------------------------------------

      TYPE ( QPB_inform_type ) :: QPB_inform
      TYPE ( QPB_pointer_type ) :: QPB_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( QPB_inform%status,                              &
                               mxGetPr( QPB_pointer%status ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%alloc_status,                        &
                               mxGetPr( QPB_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( QPB_pointer%pointer,                            &
                               'bad_alloc', QPB_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( QPB_inform%iter,                                &
                               mxGetPr( QPB_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%cg_iter,                             &
                               mxGetPr( QPB_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%factorization_status,                &
                               mxGetPr( QPB_pointer%factorization_status ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%factorization_integer,               &
                               mxGetPr( QPB_pointer%factorization_integer ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%factorization_real,                  &
                               mxGetPr( QPB_pointer%factorization_real ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%nfacts,                              &
                               mxGetPr( QPB_pointer%nfacts ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%nbacts,                              &
                               mxGetPr( QPB_pointer%nbacts ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%nmods,                               &
                               mxGetPr( QPB_pointer%nmods ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%obj,                                 &
                               mxGetPr( QPB_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%non_negligible_pivot,                &
                               mxGetPr( QPB_pointer%non_negligible_pivot ) )
      CALL MATLAB_copy_to_ptr( QPB_inform%feasible,                            &
                               mxGetPr( QPB_pointer%feasible ) )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%total, wp ),              &
                       mxGetPr( QPB_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%preprocess, wp ),         &
                       mxGetPr( QPB_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%find_dependent, wp ),     &
                       mxGetPr( QPB_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%analyse, wp ),            &
                       mxGetPr( QPB_pointer%time_pointer%analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%factorize, wp ),          &
                       mxGetPr( QPB_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%solve, wp ),              &
                       mxGetPr( QPB_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%phase1_total, wp ),       &
                       mxGetPr( QPB_pointer%time_pointer%phase1_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%phase1_analyse, wp ),     &
                      mxGetPr( QPB_pointer%time_pointer%phase1_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%phase1_factorize, wp ),   &
                      mxGetPr( QPB_pointer%time_pointer%phase1_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%phase1_solve, wp ),       &
                      mxGetPr( QPB_pointer%time_pointer%phase1_solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_total, wp ),        &
                      mxGetPr( QPB_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( QPB_pointer%time_pointer%clock_preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_find_dependent,wp), &
                      mxGetPr( QPB_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_analyse, wp ),      &
                      mxGetPr( QPB_pointer%time_pointer%clock_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_factorize, wp ),    &
                      mxGetPr( QPB_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_solve, wp ),        &
                      mxGetPr( QPB_pointer%time_pointer%clock_solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_phase1_total, wp ), &
                      mxGetPr( QPB_pointer%time_pointer%clock_phase1_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_phase1_analyse, wp),&
                      mxGetPr( QPB_pointer%time_pointer%clock_phase1_analyse ) )
      CALL MATLAB_copy_to_ptr( REAL(QPB_inform%time%clock_phase1_factorize,wp),&
                      mxGetPr( QPB_pointer%time_pointer%phase1_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( QPB_inform%time%clock_phase1_solve, wp ), &
                      mxGetPr( QPB_pointer%time_pointer%clock_phase1_solve ) )

!  initial-point calculation components

      CALL LSQP_matlab_inform_get( QPB_inform%LSQP_inform,                     &
                                   QPB_pointer%LSQP_pointer )

!  constraint-dependency check components

      CALL FDC_matlab_inform_get( QPB_inform%FDC_inform,                       &
                                  QPB_pointer%FDC_pointer )

!  linear system solver components

      CALL SBLS_matlab_inform_get( QPB_inform%SBLS_inform,                     &
                                   QPB_pointer%SBLS_pointer )

!  step computation components

      CALL GLTR_matlab_inform_get( QPB_inform%GLTR_inform,                     &
                                   QPB_pointer%GLTR_pointer )

      RETURN

!  End of subroutine QPB_matlab_inform_get

      END SUBROUTINE QPB_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ Q P B _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_QPB_MATLAB_TYPES
