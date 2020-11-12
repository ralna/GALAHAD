#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.2 - 05/03/2019 AT 15:11 GMT.

!-*-*-*-  G A L A H A D _ B S C _ M A T L A B _ T Y P E S   M O D U L E  -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.2. March 5th, 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BSC_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to BSC

      USE GALAHAD_MATLAB
      USE GALAHAD_BSC_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BSC_matlab_control_set, BSC_matlab_control_get,                &
                BSC_matlab_inform_create, BSC_matlab_inform_get

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

      TYPE, PUBLIC :: BSC_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: max_col_a, exceeds_max_col
        mwPointer :: time, clock_time
      END TYPE
    CONTAINS

!-*-  B S C _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE BSC_matlab_control_set( ps, BSC_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to BSC

!  Arguments

!  ps - given pointer to the structure
!  BSC_control - BSC control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( BSC_control_type ) :: BSC_control

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
                                 pc, BSC_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, BSC_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, BSC_control%print_level )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, BSC_control%max_col )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, BSC_control%new_a )
        CASE( 'extra_space_s' )
          CALL MATLAB_get_value( ps, 'extra_space_s',                          &
                                 pc, BSC_control%extra_space_s )
        CASE( 's_also_by_column' )
          CALL MATLAB_get_value( ps, 's_also_by_column',                       &
                                 pc, BSC_control%s_also_by_column )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, BSC_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, BSC_control%deallocate_error_fatal  )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, BSC_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine BSC_matlab_control_set

      END SUBROUTINE BSC_matlab_control_set

!-*-  B S C _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE BSC_matlab_control_get( struct, BSC_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to BSC

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BSC_control - BSC control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BSC_control_type ) :: BSC_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 10
      CHARACTER ( LEN = 25 ), PARAMETER :: finform( ninform ) = (/             &
           'error                    ', 'out                      ',           &
           'print_level              ', 'max_col                  ',           &
           'new_a                    ', 'extra_space_s            ',           &
           's_also_by_column         ', 'space_critical           ',           &
           'deallocate_error_fatal   ', 'prefix                   ' /)

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
                                  BSC_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  BSC_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  BSC_control%print_level )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  BSC_control%max_col )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  BSC_control%new_a )
      CALL MATLAB_fill_component( pointer, 'extra_space_s',                    &
                                  BSC_control%extra_space_s )
      CALL MATLAB_fill_component( pointer, 's_also_by_column',                 &
                                  BSC_control%s_also_by_column )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  BSC_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  BSC_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  BSC_control%prefix )

      RETURN

!  End of subroutine BSC_matlab_control_get

      END SUBROUTINE BSC_matlab_control_get

!-*- B S C _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE BSC_matlab_inform_create( struct, BSC_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold BSC_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  BSC_pointer - BSC pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( BSC_pointer_type ) :: BSC_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 7
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'max_col_a            ',                   &
           'exceeds_max_col      ', 'time                 ',                   &
           'clock_time           ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, BSC_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        BSC_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( BSC_pointer%pointer,               &
        'status', BSC_pointer%status )
      CALL MATLAB_create_integer_component( BSC_pointer%pointer,               &
         'alloc_status', BSC_pointer%alloc_status )
      CALL MATLAB_create_char_component( BSC_pointer%pointer,                  &
        'bad_alloc', BSC_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( BSC_pointer%pointer,               &
         'max_col_a', BSC_pointer%max_col_a )
      CALL MATLAB_create_integer_component( BSC_pointer%pointer,               &
         'exceeds_max_col', BSC_pointer%exceeds_max_col )
      CALL MATLAB_create_integer_component( BSC_pointer%pointer,               &
         'time', BSC_pointer%time )
      CALL MATLAB_create_real_component( BSC_pointer%pointer,                  &
         'clock_time', BSC_pointer%clock_time )

      RETURN

!  End of subroutine BSC_matlab_inform_create

      END SUBROUTINE BSC_matlab_inform_create

!-*-*  B S C _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   *-*-

      SUBROUTINE BSC_matlab_inform_get( BSC_inform, BSC_pointer )

!  --------------------------------------------------------------

!  Set BSC_inform values from matlab pointers

!  Arguments

!  BSC_inform - BSC inform structure
!  BSC_pointer - BSC pointer structure

!  --------------------------------------------------------------

      TYPE ( BSC_inform_type ) :: BSC_inform
      TYPE ( BSC_pointer_type ) :: BSC_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( BSC_inform%status,                              &
                               mxGetPr( BSC_pointer%status ) )
      CALL MATLAB_copy_to_ptr( BSC_inform%alloc_status,                        &
                               mxGetPr( BSC_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( BSC_pointer%pointer,                            &
                               'bad_alloc', BSC_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( BSC_inform%max_col_a,                           &
                               mxGetPr( BSC_pointer%max_col_a ) )
      CALL MATLAB_copy_to_ptr( BSC_inform%exceeds_max_col,                     &
                               mxGetPr( BSC_pointer%exceeds_max_col ) )
      CALL MATLAB_copy_to_ptr( BSC_inform%time,                                &
                               mxGetPr( BSC_pointer%time ) )
      CALL MATLAB_copy_to_ptr( BSC_inform%time,                                &
                               mxGetPr( BSC_pointer%clock_time ) )

      RETURN

!  End of subroutine BSC_matlab_inform_get

      END SUBROUTINE BSC_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ B S C _ T Y P E S   M O D U L E  -*-*-*-*-

    END MODULE GALAHAD_BSC_MATLAB_TYPES
