#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 14:45 GMT.

!-*-*-*-  G A L A H A D _ L 2 R T _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 19th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_L2RT_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to L2RT

      USE GALAHAD_MATLAB
      USE GALAHAD_L2RT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: L2RT_matlab_control_set, L2RT_matlab_control_get,              &
                L2RT_matlab_inform_create, L2RT_matlab_inform_get

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

      TYPE, PUBLIC :: L2RT_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, iter_pass2
        mwPointer :: obj, multiplier, x_norm, r_norm, Atr_norm
      END TYPE 
    CONTAINS

!-*-  L 2 R T _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE L2RT_matlab_control_set( ps, L2RT_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to L2RT

!  Arguments

!  ps - given pointer to the structure
!  L2RT_control - L2RT control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( L2RT_control_type ) :: L2RT_control

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
                                 pc, L2RT_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, L2RT_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, L2RT_control%print_level )
        CASE( 'itmin' )
          CALL MATLAB_get_value( ps, 'itmin',                                  &
                                 pc, L2RT_control%itmin )
        CASE( 'itmax' )
          CALL MATLAB_get_value( ps, 'itmax',                                  &
                                 pc, L2RT_control%itmax )
        CASE( 'bitmax' )
          CALL MATLAB_get_value( ps, 'bitmax',                                 &
                                 pc, L2RT_control%bitmax )
        CASE( 'extra_vectors' )
          CALL MATLAB_get_value( ps, 'extra_vectors',                          &
                                 pc, L2RT_control%extra_vectors )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, L2RT_control%stop_relative )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, L2RT_control%stop_absolute )
        CASE( 'fraction_opt' )
          CALL MATLAB_get_value( ps, 'fraction_opt',                           &
                                 pc, L2RT_control%fraction_opt )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, L2RT_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, L2RT_control%deallocate_error_fatal )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, L2RT_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine L2RT_matlab_control_set

      END SUBROUTINE L2RT_matlab_control_set

!-*-  L 2 R T _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE L2RT_matlab_control_get( struct, L2RT_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to L2RT

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  L2RT_control - L2RT control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( L2RT_control_type ) :: L2RT_control
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
                                  L2RT_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  L2RT_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  L2RT_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itmin',                            &
                                  L2RT_control%itmin )
      CALL MATLAB_fill_component( pointer, 'itmax',                            &
                                  L2RT_control%itmax )
      CALL MATLAB_fill_component( pointer, 'bitmax',                           &
                                  L2RT_control%bitmax )
      CALL MATLAB_fill_component( pointer, 'extra_vectors',                    &
                                  L2RT_control%extra_vectors )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  L2RT_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  L2RT_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'fraction_opt',                     &
                                  L2RT_control%fraction_opt )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  L2RT_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  L2RT_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  L2RT_control%prefix )

      RETURN

!  End of subroutine L2RT_matlab_control_get

      END SUBROUTINE L2RT_matlab_control_get

!-*- L 2 R T _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE L2RT_matlab_inform_create( struct, L2RT_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold L2RT_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  L2RT_pointer - L2RT pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( L2RT_pointer_type ) :: L2RT_pointer
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
        CALL MATLAB_create_substructure( struct, name, L2RT_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        L2RT_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( L2RT_pointer%pointer,              &
        'status', L2RT_pointer%status )
      CALL MATLAB_create_integer_component( L2RT_pointer%pointer,              &
         'alloc_status', L2RT_pointer%alloc_status )
      CALL MATLAB_create_char_component( L2RT_pointer%pointer,                 &
        'bad_alloc', L2RT_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( L2RT_pointer%pointer,              &
         'iter', L2RT_pointer%iter )
      CALL MATLAB_create_integer_component( L2RT_pointer%pointer,              &
        'iter_pass2', L2RT_pointer%iter_pass2 )
      CALL MATLAB_create_real_component( L2RT_pointer%pointer,                 &
        'obj',  L2RT_pointer%obj )
      CALL MATLAB_create_real_component( L2RT_pointer%pointer,                 &
        'multiplier',  L2RT_pointer%multiplier )
      CALL MATLAB_create_real_component( L2RT_pointer%pointer,                 &
        'x_norm',  L2RT_pointer%x_norm )
      CALL MATLAB_create_real_component( L2RT_pointer%pointer,                 &
        'r_norm',  L2RT_pointer%r_norm )
      CALL MATLAB_create_real_component( L2RT_pointer%pointer,                 &
        'Atr_norm',  L2RT_pointer%Atr_norm )

      RETURN

!  End of subroutine L2RT_matlab_inform_create

      END SUBROUTINE L2RT_matlab_inform_create

!-*-  L 2 R T _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE L2RT_matlab_inform_get( L2RT_inform, L2RT_pointer )

!  --------------------------------------------------------------

!  Set L2RT_inform values from matlab pointers

!  Arguments

!  L2RT_inform - L2RT inform structure
!  L2RT_pointer - L2RT pointer structure

!  --------------------------------------------------------------

      TYPE ( L2RT_inform_type ) :: L2RT_inform
      TYPE ( L2RT_pointer_type ) :: L2RT_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( L2RT_inform%status,                             &
                               mxGetPr( L2RT_pointer%status ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%alloc_status,                       &
                               mxGetPr( L2RT_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( L2RT_pointer%pointer,                           &
                               'bad_alloc', L2RT_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( L2RT_inform%iter,                               &
                               mxGetPr( L2RT_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%iter_pass2,                         &
                               mxGetPr( L2RT_pointer%iter_pass2 ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%obj,                                &
                               mxGetPr( L2RT_pointer%obj ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%multiplier,                         &
                               mxGetPr( L2RT_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%x_norm,                             &
                               mxGetPr( L2RT_pointer%x_norm ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%r_norm,                             &
                               mxGetPr( L2RT_pointer%r_norm ) )
      CALL MATLAB_copy_to_ptr( L2RT_inform%Atr_norm,                           &
                               mxGetPr( L2RT_pointer%Atr_norm ) )

      RETURN

!  End of subroutine L2RT_matlab_inform_get

      END SUBROUTINE L2RT_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ L 2 R T _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_L2RT_MATLAB_TYPES
