#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 11:30 GMT.

!-*-*-*-  G A L A H A D _ L S T R _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LSTR_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LSTR

      USE GALAHAD_MATLAB
      USE GALAHAD_LSTR_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LSTR_matlab_control_set, LSTR_matlab_control_get,              &
                LSTR_matlab_inform_create, LSTR_matlab_inform_get

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

      TYPE, PUBLIC :: LSTR_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, iter_pass2
        mwPointer :: multiplier, x_norm, r_norm, Atr_norm
      END TYPE 
    CONTAINS

!-*-  L S T R _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE LSTR_matlab_control_set( ps, LSTR_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LSTR

!  Arguments

!  ps - given pointer to the structure
!  LSTR_control - LSTR control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LSTR_control_type ) :: LSTR_control

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
                                 pc, LSTR_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LSTR_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LSTR_control%print_level )
        CASE( 'itmin' )
          CALL MATLAB_get_value( ps, 'itmin',                                  &
                                 pc, LSTR_control%itmin )
        CASE( 'itmax' )
          CALL MATLAB_get_value( ps, 'itmax',                                  &
                                 pc, LSTR_control%itmax )
        CASE( 'bitmax' )
          CALL MATLAB_get_value( ps, 'bitmax',                                 &
                                 pc, LSTR_control%bitmax )
        CASE( 'itmax_on_boundary' )
          CALL MATLAB_get_value( ps, 'itmax_on_boundary',                      &
                                 pc, LSTR_control%itmax_on_boundary )
        CASE( 'extra_vectors' )
          CALL MATLAB_get_value( ps, 'extra_vectors',                          &
                                 pc, LSTR_control%extra_vectors )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, LSTR_control%stop_relative )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, LSTR_control%stop_absolute )
        CASE( 'fraction_opt' )
          CALL MATLAB_get_value( ps, 'fraction_opt',                           &
                                 pc, LSTR_control%fraction_opt )
        CASE( 'steihaug_toint' )
          CALL MATLAB_get_value( ps, 'steihaug_toint',                         &
                                 pc, LSTR_control%steihaug_toint )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LSTR_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LSTR_control%deallocate_error_fatal )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LSTR_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LSTR_matlab_control_set

      END SUBROUTINE LSTR_matlab_control_set

!-*-  L S T R _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE LSTR_matlab_control_get( struct, LSTR_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LSTR

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSTR_control - LSTR control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSTR_control_type ) :: LSTR_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 15
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'itmin                          ', &
         'itmax                          ', 'bitmax                         ', &
         'itmax_on_boundary              ', 'extra_vectors                  ', &
         'stop_relative                  ', 'stop_absolute                  ', &
         'fraction_opt                   ', 'steihaug_toint                 ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         '                       /)

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
                                  LSTR_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LSTR_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LSTR_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itmin',                            &
                                  LSTR_control%itmin )
      CALL MATLAB_fill_component( pointer, 'itmax',                            &
                                  LSTR_control%itmax )
      CALL MATLAB_fill_component( pointer, 'bitmax',                           &
                                  LSTR_control%bitmax )
      CALL MATLAB_fill_component( pointer, 'itmax_on_boundary',                &
                                  LSTR_control%itmax_on_boundary )
      CALL MATLAB_fill_component( pointer, 'extra_vectors',                    &
                                  LSTR_control%extra_vectors )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  LSTR_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  LSTR_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'fraction_opt',                     &
                                  LSTR_control%fraction_opt )
      CALL MATLAB_fill_component( pointer, 'steihaug_toint',                   &
                                  LSTR_control%steihaug_toint )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LSTR_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LSTR_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LSTR_control%prefix )

      RETURN

!  End of subroutine LSTR_matlab_control_get

      END SUBROUTINE LSTR_matlab_control_get

!-*- L S T R _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE LSTR_matlab_inform_create( struct, LSTR_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LSTR_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LSTR_pointer - LSTR pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LSTR_pointer_type ) :: LSTR_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'iter_pass2           ', 'multiplier           ',                   &
           'x_norm               ', 'r_norm               ',                   &
           'Atr_norm             '                               /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LSTR_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        LSTR_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( LSTR_pointer%pointer,              &
        'status', LSTR_pointer%status )
      CALL MATLAB_create_integer_component( LSTR_pointer%pointer,              &
         'alloc_status', LSTR_pointer%alloc_status )
      CALL MATLAB_create_char_component( LSTR_pointer%pointer,                 &
        'bad_alloc', LSTR_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( LSTR_pointer%pointer,              &
         'iter', LSTR_pointer%iter )
      CALL MATLAB_create_integer_component( LSTR_pointer%pointer,              &
        'iter_pass2', LSTR_pointer%iter_pass2 )
      CALL MATLAB_create_real_component( LSTR_pointer%pointer,                 &
        'multiplier',  LSTR_pointer%multiplier )
      CALL MATLAB_create_real_component( LSTR_pointer%pointer,                 &
        'x_norm',  LSTR_pointer%x_norm )
      CALL MATLAB_create_real_component( LSTR_pointer%pointer,                 &
        'r_norm',  LSTR_pointer%r_norm )
      CALL MATLAB_create_real_component( LSTR_pointer%pointer,                 &
        'Atr_norm',  LSTR_pointer%Atr_norm )

      RETURN

!  End of subroutine LSTR_matlab_inform_create

      END SUBROUTINE LSTR_matlab_inform_create

!-*-  L S T R _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE LSTR_matlab_inform_get( LSTR_inform, LSTR_pointer )

!  --------------------------------------------------------------

!  Set LSTR_inform values from matlab pointers

!  Arguments

!  LSTR_inform - LSTR inform structure
!  LSTR_pointer - LSTR pointer structure

!  --------------------------------------------------------------

      TYPE ( LSTR_inform_type ) :: LSTR_inform
      TYPE ( LSTR_pointer_type ) :: LSTR_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LSTR_inform%status,                             &
                               mxGetPr( LSTR_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%alloc_status,                       &
                               mxGetPr( LSTR_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LSTR_pointer%pointer,                           &
                               'bad_alloc', LSTR_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( LSTR_inform%iter,                               &
                               mxGetPr( LSTR_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%iter_pass2,                         &
                               mxGetPr( LSTR_pointer%iter_pass2 ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%multiplier,                         &
                               mxGetPr( LSTR_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%x_norm,                             &
                               mxGetPr( LSTR_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%r_norm,                             &
                               mxGetPr( LSTR_pointer%r_norm ) )
      CALL MATLAB_copy_to_ptr( LSTR_inform%Atr_norm,                           &
                               mxGetPr( LSTR_pointer%Atr_norm ) )

      RETURN

!  End of subroutine LSTR_matlab_inform_get

      END SUBROUTINE LSTR_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L S T R _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LSTR_MATLAB_TYPES
