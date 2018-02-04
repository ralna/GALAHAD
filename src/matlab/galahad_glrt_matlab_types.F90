#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 26/02/2010 AT 14:30 GMT.

!-*-*-*-  G A L A H A D _ G L R T _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_GLRT_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to GLRT

      USE GALAHAD_MATLAB
      USE GALAHAD_GLRT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: GLRT_matlab_control_set, GLRT_matlab_control_get,              &
                GLRT_matlab_inform_create, GLRT_matlab_inform_get

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

      TYPE, PUBLIC :: GLRT_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, iter_pass2
        mwPointer :: obj, obj_regularized, multiplier, xpo_norm, leftmost
        mwPointer :: negative_curvature, hard_case
      END TYPE
    CONTAINS

!-*-  G L R T _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE GLRT_matlab_control_set( ps, GLRT_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to GLRT

!  Arguments

!  ps - given pointer to the structure
!  GLRT_control - GLRT control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( GLRT_control_type ) :: GLRT_control

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
                                 pc, GLRT_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, GLRT_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, GLRT_control%print_level )
        CASE( 'itmax' )
          CALL MATLAB_get_value( ps, 'itmax',                                  &
                                 pc, GLRT_control%itmax )
        CASE( 'stopping_rule' )
          CALL MATLAB_get_value( ps, 'stopping_rule',                          &
                                 pc, GLRT_control%stopping_rule )
        CASE( 'freq' )
          CALL MATLAB_get_value( ps, 'freq',                                   &
                                 pc, GLRT_control%freq )
        CASE( 'extra_vectors' )
          CALL MATLAB_get_value( ps, 'extra_vectors',                          &
                                 pc, GLRT_control%extra_vectors )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, GLRT_control%stop_relative )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, GLRT_control%stop_absolute )
        CASE( 'fraction_opt' )
          CALL MATLAB_get_value( ps, 'fraction_opt',                           &
                                 pc, GLRT_control%fraction_opt )
        CASE( 'rminvr_zero' )
          CALL MATLAB_get_value( ps, 'rminvr_zero',                            &
                                 pc, GLRT_control%rminvr_zero )
        CASE( 'f_0' )
          CALL MATLAB_get_value( ps, 'f_0',                                    &
                                 pc, GLRT_control%f_0 )
        CASE( 'unitm' )
          CALL MATLAB_get_value( ps, 'unitm',                                  &
                                 pc, GLRT_control%unitm )
        CASE( 'impose_descent' )
          CALL MATLAB_get_value( ps, 'impose_descent',                         &
                                 pc, GLRT_control%impose_descent )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, GLRT_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, GLRT_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, GLRT_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine GLRT_matlab_control_set

      END SUBROUTINE GLRT_matlab_control_set

!-*-  G L T R _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE GLRT_matlab_control_get( struct, GLRT_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to GLRT

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  GLRT_control - GLRT control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( GLRT_control_type ) :: GLRT_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 17
      CHARACTER ( LEN = 22 ), PARAMETER :: finform( ninform ) = (/             &
           'error                 ', 'out                   ',                 &
           'print_level           ', 'itmax                 ',                 &
           'stopping_rule         ', 'freq                  ',                 &
           'extra_vectors         ', 'stop_relative         ',                 &
           'stop_absolute         ', 'fraction_opt          ',                 &
           'rminvr_zero           ', 'f_0                   ',                 &
           'unitm                 ', 'impose_descent        ',                 &
           'space_critical        ', 'deallocate_error_fatal',                 &
           'prefix                '                              /)

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
                                  GLRT_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  GLRT_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  GLRT_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itmax',                            &
                                  GLRT_control%itmax )
      CALL MATLAB_fill_component( pointer, 'stopping_rule',                    &
                                  GLRT_control%stopping_rule )
      CALL MATLAB_fill_component( pointer, 'freq',                             &
                                  GLRT_control%freq )
      CALL MATLAB_fill_component( pointer, 'extra_vectors',                    &
                                  GLRT_control%extra_vectors )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  GLRT_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  GLRT_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'fraction_opt',                     &
                                  GLRT_control%fraction_opt )
      CALL MATLAB_fill_component( pointer, 'rminvr_zero',                      &
                                  GLRT_control%rminvr_zero )
      CALL MATLAB_fill_component( pointer, 'f_0',                              &
                                  GLRT_control%f_0 )
      CALL MATLAB_fill_component( pointer, 'unitm',                            &
                                  GLRT_control%unitm )
      CALL MATLAB_fill_component( pointer, 'impose_descent',                   &
                                  GLRT_control%impose_descent )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  GLRT_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  GLRT_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  GLRT_control%prefix )

      RETURN

!  End of subroutine GLRT_matlab_control_get

      END SUBROUTINE GLRT_matlab_control_get

!-*- G L R T _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE GLRT_matlab_inform_create( struct, GLRT_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold GLRT_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  GLRT_pointer - GLRT pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( GLRT_pointer_type ) :: GLRT_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 12
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'iter_pass2           ', 'obj                  ',                   &
           'obj_regularized      ', 'multiplier           ',                   &
           'xpo_norm             ', 'leftmost             ',                   &
           'negative_curvature   ', 'hard_case            '     /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, GLRT_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        GLRT_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( GLRT_pointer%pointer,              &
        'status', GLRT_pointer%status )
      CALL MATLAB_create_integer_component( GLRT_pointer%pointer,              &
         'alloc_status', GLRT_pointer%alloc_status )
      CALL MATLAB_create_char_component( GLRT_pointer%pointer,                 &
        'bad_alloc', GLRT_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( GLRT_pointer%pointer,              &
         'iter', GLRT_pointer%iter )
      CALL MATLAB_create_integer_component( GLRT_pointer%pointer,              &
        'iter_pass2', GLRT_pointer%iter_pass2 )
      CALL MATLAB_create_real_component( GLRT_pointer%pointer,                 &
        'obj',  GLRT_pointer%obj )
      CALL MATLAB_create_real_component( GLRT_pointer%pointer,                 &
        'obj_regularized',  GLRT_pointer%obj_regularized )
      CALL MATLAB_create_real_component( GLRT_pointer%pointer,                 &
        'multiplier',  GLRT_pointer%multiplier )
      CALL MATLAB_create_real_component( GLRT_pointer%pointer,                 &
        'xpo_norm',  GLRT_pointer%xpo_norm )
      CALL MATLAB_create_real_component( GLRT_pointer%pointer,                 &
        'leftmost',  GLRT_pointer%leftmost )
      CALL MATLAB_create_logical_component( GLRT_pointer%pointer,              &
        'negative_curvature', GLRT_pointer%negative_curvature )
      CALL MATLAB_create_logical_component( GLRT_pointer%pointer,              &
        'hard_case', GLRT_pointer%hard_case )

      RETURN

!  End of subroutine GLRT_matlab_inform_create

      END SUBROUTINE GLRT_matlab_inform_create

!-*-  G L R T _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE GLRT_matlab_inform_get( GLRT_inform, GLRT_pointer )

!  --------------------------------------------------------------

!  Set GLRT_inform values from matlab pointers

!  Arguments

!  GLRT_inform - GLRT inform structure
!  GLRT_pointer - GLRT pointer structure

!  --------------------------------------------------------------

      TYPE ( GLRT_inform_type ) :: GLRT_inform
      TYPE ( GLRT_pointer_type ) :: GLRT_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( GLRT_inform%status,                             &
                               mxGetPr( GLRT_pointer%status ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%alloc_status,                       &
                               mxGetPr( GLRT_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( GLRT_pointer%pointer,                           &
                               'bad_alloc', GLRT_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( GLRT_inform%iter,                               &
                               mxGetPr( GLRT_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%iter_pass2,                         &
                               mxGetPr( GLRT_pointer%iter_pass2 ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%obj,                                &
                               mxGetPr( GLRT_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%obj_regularized,                 &
                               mxGetPr( GLRT_pointer%obj_regularized ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%multiplier,                         &
                               mxGetPr( GLRT_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%xpo_norm,                           &
                               mxGetPr( GLRT_pointer%xpo_norm ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%leftmost,                           &
                               mxGetPr( GLRT_pointer%leftmost ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%negative_curvature,                 &
                               mxGetPr( GLRT_pointer%negative_curvature ) )
      CALL MATLAB_copy_to_ptr( GLRT_inform%hard_case,                          &
                               mxGetPr( GLRT_pointer%hard_case ) )

      RETURN

!  End of subroutine GLRT_matlab_inform_get

      END SUBROUTINE GLRT_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ G L R T _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_GLRT_MATLAB_TYPES
