#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 14:45 GMT.

!-*-*-*-  G A L A H A D _ L S R T _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LSRT_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LSRT

      USE GALAHAD_MATLAB
      USE GALAHAD_LSRT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LSRT_matlab_control_set, LSRT_matlab_control_get,              &
                LSRT_matlab_inform_create, LSRT_matlab_inform_get

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

      TYPE, PUBLIC :: LSRT_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, iter_pass2
        mwPointer :: obj, multiplier, x_norm, r_norm, Atr_norm
      END TYPE 
    CONTAINS

!-*-  L S R T _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE LSRT_matlab_control_set( ps, LSRT_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LSRT

!  Arguments

!  ps - given pointer to the structure
!  LSRT_control - LSRT control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LSRT_control_type ) :: LSRT_control

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
                                 pc, LSRT_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LSRT_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LSRT_control%print_level )
        CASE( 'itmin' )
          CALL MATLAB_get_value( ps, 'itmin',                                  &
                                 pc, LSRT_control%itmin )
        CASE( 'itmax' )
          CALL MATLAB_get_value( ps, 'itmax',                                  &
                                 pc, LSRT_control%itmax )
        CASE( 'bitmax' )
          CALL MATLAB_get_value( ps, 'bitmax',                                 &
                                 pc, LSRT_control%bitmax )
        CASE( 'extra_vectors' )
          CALL MATLAB_get_value( ps, 'extra_vectors',                          &
                                 pc, LSRT_control%extra_vectors )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, LSRT_control%stop_relative )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, LSRT_control%stop_absolute )
        CASE( 'fraction_opt' )
          CALL MATLAB_get_value( ps, 'fraction_opt',                           &
                                 pc, LSRT_control%fraction_opt )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LSRT_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LSRT_control%deallocate_error_fatal )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LSRT_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LSRT_matlab_control_set

      END SUBROUTINE LSRT_matlab_control_set

!-*-  L S R T _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LSRT_matlab_control_get( struct, LSRT_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LSRT

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSRT_control - LSRT control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSRT_control_type ) :: LSRT_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 13
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'itmin                          ', &
         'itmax                          ', 'bitmax                         ', &
         'extra_vectors                  ', 'stop_relative                  ', &
         'stop_absolute                  ', 'fraction_opt                   ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         '                        /)

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
                                  LSRT_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LSRT_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LSRT_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itmin',                            &
                                  LSRT_control%itmin )
      CALL MATLAB_fill_component( pointer, 'itmax',                            &
                                  LSRT_control%itmax )
      CALL MATLAB_fill_component( pointer, 'bitmax',                           &
                                  LSRT_control%bitmax )
      CALL MATLAB_fill_component( pointer, 'extra_vectors',                    &
                                  LSRT_control%extra_vectors )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  LSRT_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  LSRT_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'fraction_opt',                     &
                                  LSRT_control%fraction_opt )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LSRT_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LSRT_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LSRT_control%prefix )

      RETURN

!  End of subroutine LSRT_matlab_control_get

      END SUBROUTINE LSRT_matlab_control_get

!-*- L S R T _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE LSRT_matlab_inform_create( struct, LSRT_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LSRT_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSRT_pointer - LSRT pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSRT_pointer_type ) :: LSRT_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 10
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'iter_pass2           ', 'obj                  ',                   &
           'multiplier           ', 'x_norm               ',                   &
           'r_norm               ', 'Atr_norm             '      /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LSRT_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LSRT_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LSRT_pointer%pointer,              &
        'status', LSRT_pointer%status )
      CALL MATLAB_create_integer_component( LSRT_pointer%pointer,              &
         'alloc_status', LSRT_pointer%alloc_status )
      CALL MATLAB_create_char_component( LSRT_pointer%pointer,                 &
        'bad_alloc', LSRT_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LSRT_pointer%pointer,              &
         'iter', LSRT_pointer%iter )
      CALL MATLAB_create_integer_component( LSRT_pointer%pointer,              &
        'iter_pass2', LSRT_pointer%iter_pass2 )
      CALL MATLAB_create_real_component( LSRT_pointer%pointer,                 &
        'obj',  LSRT_pointer%obj )
      CALL MATLAB_create_real_component( LSRT_pointer%pointer,                 &
        'multiplier',  LSRT_pointer%multiplier )
      CALL MATLAB_create_real_component( LSRT_pointer%pointer,                 &
        'x_norm',  LSRT_pointer%x_norm )
      CALL MATLAB_create_real_component( LSRT_pointer%pointer,                 &
        'r_norm',  LSRT_pointer%r_norm )
      CALL MATLAB_create_real_component( LSRT_pointer%pointer,                 &
        'Atr_norm',  LSRT_pointer%Atr_norm )

      RETURN

!  End of subroutine LSRT_matlab_inform_create

      END SUBROUTINE LSRT_matlab_inform_create

!-*-  L S R T _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE LSRT_matlab_inform_get( LSRT_inform, LSRT_pointer )

!  --------------------------------------------------------------

!  Set LSRT_inform values from matlab pointers

!  Arguments

!  LSRT_inform - LSRT inform structure
!  LSRT_pointer - LSRT pointer structure

!  --------------------------------------------------------------

      TYPE ( LSRT_inform_type ) :: LSRT_inform
      TYPE ( LSRT_pointer_type ) :: LSRT_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LSRT_inform%status,                             &
                               mxGetPr( LSRT_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%alloc_status,                       &
                               mxGetPr( LSRT_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LSRT_pointer%pointer,                           &
                               'bad_alloc', LSRT_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LSRT_inform%iter,                               &
                               mxGetPr( LSRT_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%iter_pass2,                         &
                               mxGetPr( LSRT_pointer%iter_pass2 ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%obj,                                &
                               mxGetPr( LSRT_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%multiplier,                         &
                               mxGetPr( LSRT_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%x_norm,                             &
                               mxGetPr( LSRT_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%r_norm,                             &
                               mxGetPr( LSRT_pointer%r_norm ) )
      CALL MATLAB_copy_to_ptr( LSRT_inform%Atr_norm,                           &
                               mxGetPr( LSRT_pointer%Atr_norm ) )

      RETURN

!  End of subroutine LSRT_matlab_inform_get

      END SUBROUTINE LSRT_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L S R T _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LSRT_MATLAB_TYPES
