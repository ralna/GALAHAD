#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 19/10/2018 AT 13:35 GMT.

!-*-*-*-  G A L A H A D _  L P A _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.1. October 19th, 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LPA_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LPA

      USE GALAHAD_MATLAB
      USE GALAHAD_LPA_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPA_matlab_control_set, LPA_matlab_control_get,                &
                LPA_matlab_inform_create, LPA_matlab_inform_get

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

      TYPE, PUBLIC :: LPA_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, preprocess
        mwPointer :: clock_total, clock_preprocess
      END TYPE

      TYPE, PUBLIC :: LPA_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, iter, la04_job
        mwPointer :: la04_job_info, obj, primal_infeasibility, feasible
!       mwPointer :: RINFO( 40 )
        TYPE ( LPA_time_pointer_type ) :: time_pointer
      END TYPE
    CONTAINS

!-*-   L P A _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE LPA_matlab_control_set( ps, LPA_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LPA

!  Arguments

!  ps - given pointer to the structure
!  LPA_control - LPA control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LPA_control_type ) :: LPA_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, LPA_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LPA_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LPA_control%print_level )
        CASE( 'start_print' )
          CALL MATLAB_get_value( ps, 'start_print',                            &
                                 pc, LPA_control%start_print )
        CASE( 'stop_print' )
          CALL MATLAB_get_value( ps, 'stop_print',                             &
                                 pc, LPA_control%stop_print )
        CASE( 'maxit' )
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, LPA_control%maxit  )
        CASE( 'max_iterative_refinements' )
          CALL MATLAB_get_value( ps, 'max_iterative_refinements',              &
                                 pc, LPA_control%max_iterative_refinements )
        CASE( 'min_real_factor_size' )
          CALL MATLAB_get_value( ps, 'min_real_factor_size',                   &
                                 pc, LPA_control%min_real_factor_size )
        CASE( 'min_integer_factor_size' )
          CALL MATLAB_get_value( ps, 'min_integer_factor_size',                &
                                 pc, LPA_control%min_integer_factor_size )
        CASE( 'random_number_seed ' )
          CALL MATLAB_get_value( ps, 'random_number_seed',                     &
                                 pc, LPA_control%random_number_seed  )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, LPA_control%infinity )
        CASE( 'tol_data' )
          CALL MATLAB_get_value( ps, 'tol_data',                               &
                                 pc, LPA_control%tol_data )
        CASE( 'feas_tol' )
          CALL MATLAB_get_value( ps, 'feas_tol',                               &
                                 pc, LPA_control%feas_tol )
        CASE( 'relative_pivot_tolerance' )
          CALL MATLAB_get_value( ps, 'relative_pivot_tolerance',               &
                                 pc, LPA_control%relative_pivot_tolerance )
        CASE( 'growth_limit' )
          CALL MATLAB_get_value( ps, 'growth_limit',                           &
                                 pc, LPA_control%growth_limit )
        CASE( 'zero_tolerance' )
          CALL MATLAB_get_value( ps, 'zero_tolerance',                         &
                                 pc, LPA_control%zero_tolerance )
        CASE( 'change_tolerance' )
          CALL MATLAB_get_value( ps, 'change_tolerance',                       &
                                 pc, LPA_control%change_tolerance )
        CASE( 'identical_bounds_tol' )
          CALL MATLAB_get_value( ps, 'identical_bounds_tol',                   &
                                 pc, LPA_control%identical_bounds_tol )
        CASE( 'cpu_time_limit' )
          CALL MATLAB_get_value( ps, 'cpu_time_limit',                         &
                                 pc, LPA_control%cpu_time_limit )
        CASE( 'clock_time_limit' )
          CALL MATLAB_get_value( ps, 'clock_time_limit',                       &
                                 pc, LPA_control%clock_time_limit )
        CASE( 'scale' )
          CALL MATLAB_get_value( ps, 'scale',                                  &
                                 pc, LPA_control%scale )
        CASE( 'dual' )
          CALL MATLAB_get_value( ps, 'dual',                                   &
                                 pc, LPA_control%dual )
        CASE( 'warm_start' )
          CALL MATLAB_get_value( ps, 'warm_start',                             &
                                 pc, LPA_control%warm_start )
        CASE( 'steepest_edge' )
          CALL MATLAB_get_value( ps, 'steepest_edge',                          &
                                 pc, LPA_control%steepest_edge )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LPA_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LPA_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LPA_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LPA_matlab_control_set

      END SUBROUTINE LPA_matlab_control_set

!-*-   L P A _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LPA_matlab_control_get( struct, LPA_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LPA

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LPA_control - LPA control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LPA_control_type ) :: LPA_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 27
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'start_print                    ', &
         'stop_print                     ', 'maxit                          ', &
         'max_iterative_refinements      ', 'min_integer_factor_size        ', &
         'min_real_factor_size           ', 'random_number_seed             ', &
         'infinity                       ', 'tol_data                       ', &
         'feas_tol                       ', 'relative_pivot_tolerance       ', &
         'growth_limit                   ', 'zero_tolerance                 ', &
         'change_tolerance               ', 'identical_bounds_tol           ', &
         'cpu_time_limit                 ', 'clock_time_limit               ', &
         'scale                          ', 'dual                           ', &
         'warm_start                     ', 'steepest_edge                  ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         ' /)

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
                                  LPA_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LPA_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LPA_control%print_level )
      CALL MATLAB_fill_component( pointer, 'start_print',                      &
                                  LPA_control%start_print )
      CALL MATLAB_fill_component( pointer, 'stop_print',                       &
                                  LPA_control%stop_print )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  LPA_control%maxit )
      CALL MATLAB_fill_component( pointer, 'max_iterative_refinements',        &
                                  LPA_control%max_iterative_refinements )
      CALL MATLAB_fill_component( pointer, 'min_real_factor_size',             &
                                  LPA_control%min_real_factor_size )
      CALL MATLAB_fill_component( pointer, 'min_integer_factor_size',          &
                                  LPA_control%min_integer_factor_size )
      CALL MATLAB_fill_component( pointer, 'random_number_seed',               &
                                  LPA_control%random_number_seed )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  LPA_control%infinity )
      CALL MATLAB_fill_component( pointer, 'tol_data',                         &
                                  LPA_control%tol_data )
      CALL MATLAB_fill_component( pointer, 'feas_tol',                         &
                                  LPA_control%feas_tol )
      CALL MATLAB_fill_component( pointer, 'relative_pivot_tolerance',         &
                                  LPA_control%relative_pivot_tolerance )
      CALL MATLAB_fill_component( pointer, 'growth_limit',                     &
                                  LPA_control%growth_limit )
      CALL MATLAB_fill_component( pointer, 'zero_tolerance',                   &
                                  LPA_control%zero_tolerance )
      CALL MATLAB_fill_component( pointer, 'change_tolerance',                 &
                                  LPA_control%change_tolerance )
      CALL MATLAB_fill_component( pointer, 'identical_bounds_tol',             &
                                  LPA_control%identical_bounds_tol )
      CALL MATLAB_fill_component( pointer, 'cpu_time_limit',                   &
                                  LPA_control%cpu_time_limit )
      CALL MATLAB_fill_component( pointer, 'clock_time_limit',                 &
                                  LPA_control%clock_time_limit )
      CALL MATLAB_fill_component( pointer, 'scale',                            &
                                  LPA_control%scale )
      CALL MATLAB_fill_component( pointer, 'dual',                             &
                                  LPA_control%dual )
      CALL MATLAB_fill_component( pointer, 'warm_start',                       &
                                  LPA_control%warm_start )
      CALL MATLAB_fill_component( pointer, 'steepest_edge',                    &
                                  LPA_control%steepest_edge )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LPA_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LPA_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LPA_control%prefix )

      RETURN

!  End of subroutine LPA_matlab_control_get

      END SUBROUTINE LPA_matlab_control_get

!-*-  L P A _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE LPA_matlab_inform_create( struct, LPA_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LPA_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LPA_pointer - LPA pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LPA_pointer_type ) :: LPA_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 10
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'iter                    ',             &
           'la04_job                ', 'la04_job_info           ',             &
           'obj                     ', 'primal_infeasibility    ',             &
           'feasible                ', 'time                    ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 4
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'preprocess           ',                   &
           'clock_total          ', 'clock_preprocess     '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LPA_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LPA_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LPA_pointer%pointer,               &
        'status', LPA_pointer%status )
      CALL MATLAB_create_integer_component( LPA_pointer%pointer,               &
         'alloc_status', LPA_pointer%alloc_status )
      CALL MATLAB_create_char_component( LPA_pointer%pointer,                  &
        'bad_alloc', LPA_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LPA_pointer%pointer,               &
        'iter', LPA_pointer%iter )
      CALL MATLAB_create_integer_component( LPA_pointer%pointer,               &
        'la04_job', LPA_pointer%la04_job )
      CALL MATLAB_create_integer_component( LPA_pointer%pointer,               &
        'la04_job_info', LPA_pointer%la04_job_info )
      CALL MATLAB_create_real_component( LPA_pointer%pointer,                  &
        'obj', LPA_pointer%obj )
      CALL MATLAB_create_real_component( LPA_pointer%pointer,                  &
         'primal_infeasibility', LPA_pointer%primal_infeasibility )
      CALL MATLAB_create_logical_component( LPA_pointer%pointer,               &
        'feasible', LPA_pointer%feasible )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( LPA_pointer%pointer,                    &
        'time', LPA_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( LPA_pointer%time_pointer%pointer,     &
        'total', LPA_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( LPA_pointer%time_pointer%pointer,     &
        'preprocess', LPA_pointer%time_pointer%preprocess )
      CALL MATLAB_create_real_component( LPA_pointer%time_pointer%pointer,     &
        'clock_total', LPA_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( LPA_pointer%time_pointer%pointer,     &
        'clock_preprocess', LPA_pointer%time_pointer%clock_preprocess )

      RETURN

!  End of subroutine LPA_matlab_inform_create

      END SUBROUTINE LPA_matlab_inform_create

!-*-*-   L P A _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE LPA_matlab_inform_get( LPA_inform, LPA_pointer )

!  --------------------------------------------------------------

!  Set LPA_inform values from matlab pointers

!  Arguments

!  LPA_inform - LPA inform structure
!  LPA_pointer - LPA pointer structure

!  --------------------------------------------------------------

      TYPE ( LPA_inform_type ) :: LPA_inform
      TYPE ( LPA_pointer_type ) :: LPA_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LPA_inform%status,                              &
                               mxGetPr( LPA_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%alloc_status,                        &
                               mxGetPr( LPA_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LPA_pointer%pointer,                            &
                               'bad_alloc', LPA_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LPA_inform%iter,                                &
                               mxGetPr( LPA_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%la04_job,                            &
                               mxGetPr( LPA_pointer%la04_job ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%la04_job_info,                       &
                               mxGetPr( LPA_pointer%la04_job_info ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%obj,                                 &
                               mxGetPr( LPA_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%primal_infeasibility,                &
                               mxGetPr( LPA_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( LPA_inform%feasible,                            &
                               mxGetPr( LPA_pointer%feasible ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( LPA_inform%time%total, wp ),              &
                               mxGetPr( LPA_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( LPA_inform%time%preprocess, wp ),         &
                               mxGetPr( LPA_pointer%time_pointer%preprocess ) )
      CALL MATLAB_copy_to_ptr( REAL( LPA_inform%time%clock_total, wp ),        &
                      mxGetPr( LPA_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( LPA_inform%time%clock_preprocess, wp ),   &
                      mxGetPr( LPA_pointer%time_pointer%clock_preprocess ) )

      RETURN

!  End of subroutine LPA_matlab_inform_get

      END SUBROUTINE LPA_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _  L P A _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LPA_MATLAB_TYPES
