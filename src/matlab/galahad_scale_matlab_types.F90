#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 01/02/2011 AT 18:30 GMT.

!-*-*-  G A L A H A D _ S C A L E _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 1st, 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SCALE_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SCALE

      USE GALAHAD_MATLAB
      USE GALAHAD_SCALE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SCALE_matlab_control_set, SCALE_matlab_control_get,            &
                SCALE_matlab_inform_create, SCALE_matlab_inform_get

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

      TYPE, PUBLIC :: SCALE_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, deviation
      END TYPE 
    CONTAINS

!-*-  S C A L E _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE SCALE_matlab_control_set( ps, SCALE_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QP

!  Arguments

!  ps - given pointer to the structure
!  SCALE_control - SCALE control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SCALE_control_type ) :: SCALE_control

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
                                 pc, SCALE_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SCALE_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SCALE_control%print_level )
        CASE( 'maxit' )                                                  
          CALL MATLAB_get_value( ps, 'maxit',                                  &
                                 pc, SCALE_control%maxit )
        CASE( 'shift_x' )                                                  
          CALL MATLAB_get_value( ps, 'shift_x',                                &
                                 pc, SCALE_control%shift_x )
        CASE( 'scale_x' )                                                  
          CALL MATLAB_get_value( ps, 'scale_x',                                &
                                 pc, SCALE_control%scale_x )
        CASE( 'shift_c' )                                                  
          CALL MATLAB_get_value( ps, 'shift_c',                                &
                                 pc, SCALE_control%shift_c )
        CASE( 'scale_c' )                                                  
          CALL MATLAB_get_value( ps, 'scale_c',                                &
                                 pc, SCALE_control%scale_c )
        CASE( 'shift_f' )                                                  
          CALL MATLAB_get_value( ps, 'shift_f',                                &
                                 pc, SCALE_control%shift_f )
        CASE( 'scale_f' )                                                  
          CALL MATLAB_get_value( ps, 'scale_f',                                &
                                 pc, SCALE_control%scale_f )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, SCALE_control%infinity )
        CASE( 'stop_tol' )
          CALL MATLAB_get_value( ps, 'stop_tol',                               &
                                 pc, SCALE_control%stop_tol )
        CASE( 'scale_x_min' )
          CALL MATLAB_get_value( ps, 'scale_x_min',                            &
                                 pc, SCALE_control%scale_x_min )
        CASE( 'scale_c_min' )
          CALL MATLAB_get_value( ps, 'scale_c_min',                            &
                                 pc, SCALE_control%scale_c_min )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SCALE_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, SCALE_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SCALE_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SCALE_matlab_control_set

      END SUBROUTINE SCALE_matlab_control_set

!-*-  S C A L E _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE SCALE_matlab_control_get( struct, SCALE_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SCALE_control - SCALE control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SCALE_control_type ) :: SCALE_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 17
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'maxit                          ', &
         'shift_x                        ', 'scale_x                        ', &
         'shift_c                        ', 'scale_c                        ', &
         'shift_f                        ', 'scale_f                        ', &
         'infinity                       ', 'stop_tol                       ', &
         'scale_x_min                    ', 'scale_c_min                    ', &
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
                                  SCALE_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SCALE_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SCALE_control%print_level )
      CALL MATLAB_fill_component( pointer, 'maxit',                            &
                                  SCALE_control%maxit )
      CALL MATLAB_fill_component( pointer, 'shift_x',                          &
                                  SCALE_control%shift_x )
      CALL MATLAB_fill_component( pointer, 'scale_x',                          &
                                  SCALE_control%scale_x )
      CALL MATLAB_fill_component( pointer, 'shift_c',                          &
                                  SCALE_control%shift_c )
      CALL MATLAB_fill_component( pointer, 'scale_c',                          &
                                  SCALE_control%scale_c )
      CALL MATLAB_fill_component( pointer, 'shift_f',                          &
                                  SCALE_control%shift_f )
      CALL MATLAB_fill_component( pointer, 'scale_f',                          &
                                  SCALE_control%scale_f )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  SCALE_control%infinity )
      CALL MATLAB_fill_component( pointer, 'stop_tol',                         &
                                  SCALE_control%stop_tol )
      CALL MATLAB_fill_component( pointer, 'scale_x_min',                      &
                                  SCALE_control%scale_x_min )
      CALL MATLAB_fill_component( pointer, 'scale_c_min',                      &
                                  SCALE_control%scale_c_min )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SCALE_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  SCALE_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SCALE_control%prefix )
      RETURN

!  End of subroutine SCALE_matlab_control_get

      END SUBROUTINE SCALE_matlab_control_get

!-* S C A L E _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  *-

      SUBROUTINE SCALE_matlab_inform_create( struct, SCALE_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SCALE_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SCALE_pointer - SCALE pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SCALE_pointer_type ) :: SCALE_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 5
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'iter                    ',             &
           'deviation               ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SCALE_pointer%pointer,  &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SCALE_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( SCALE_pointer%pointer,             &
        'status', SCALE_pointer%status )
      CALL MATLAB_create_integer_component( SCALE_pointer%pointer,             &
         'alloc_status', SCALE_pointer%alloc_status )
      CALL MATLAB_create_char_component( SCALE_pointer%pointer,                &
        'bad_alloc', SCALE_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( SCALE_pointer%pointer,             &
         'iter', SCALE_pointer%iter )
      CALL MATLAB_create_real_component( SCALE_pointer%pointer,                &
        'deviation', SCALE_pointer%deviation )

      RETURN

!  End of subroutine SCALE_matlab_inform_create

      END SUBROUTINE SCALE_matlab_inform_create

!-*-  S C A L E _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE SCALE_matlab_inform_get( SCALE_inform, SCALE_pointer )

!  --------------------------------------------------------------

!  Set SCALE_inform values from matlab pointers

!  Arguments

!  SCALE_inform - SCALE inform structure
!  SCALE_pointer - SCALE pointer structure

!  --------------------------------------------------------------

      TYPE ( SCALE_inform_type ) :: SCALE_inform
      TYPE ( SCALE_pointer_type ) :: SCALE_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SCALE_inform%status,                            &
                               mxGetPr( SCALE_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SCALE_inform%alloc_status,                      &
                               mxGetPr( SCALE_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SCALE_pointer%pointer,                          &
                               'bad_alloc', SCALE_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( SCALE_inform%iter,                              &
                               mxGetPr( SCALE_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( SCALE_inform%deviation,                         &
                               mxGetPr( SCALE_pointer%deviation ) )

      RETURN

!  End of subroutine SCALE_matlab_inform_get

      END SUBROUTINE SCALE_matlab_inform_get

!-*-*-  E N D  o f  G A L A H A D _ S C A L E _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SCALE_MATLAB_TYPES
