#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 09:40 GMT.

!-*-*-*-  G A L A H A D _ H A S H _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 14th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_HASH_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to HASH

      USE GALAHAD_MATLAB
      USE GALAHAD_HASH

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: HASH_matlab_control_set, HASH_matlab_control_get,              &
                HASH_matlab_inform_create, HASH_matlab_inform_get

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

      TYPE, PUBLIC :: HASH_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
      END TYPE

    CONTAINS

!-*-  H A S H _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E  -*-

      SUBROUTINE HASH_matlab_control_set( ps, HASH_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to HASH

!  Arguments

!  ps - given pointer to the structure
!  HASH_control - HASH control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( HASH_control_type ) :: HASH_control

!  local variables

      INTEGER :: i, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO i = 1, nfields
        name = mxGetFieldNameByNumber( ps, i )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, HASH_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, HASH_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, HASH_control%print_level )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, HASH_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, HASH_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, HASH_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine HASH_matlab_control_set

      END SUBROUTINE HASH_matlab_control_set

!-*-  H A S H _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE HASH_matlab_control_get( struct, HASH_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to HASH

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  HASH_control - HASH control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( HASH_control_type ) :: HASH_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 6
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'space_critical                 ', &
         'deallocate_error_fatal         ', 'prefix                         ' /)

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
                                  HASH_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  HASH_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  HASH_control%print_level )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  HASH_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  HASH_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  HASH_control%prefix )

      RETURN

!  End of subroutine HASH_matlab_control_get

      END SUBROUTINE HASH_matlab_control_get

!-*- H A S H _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E -*-

      SUBROUTINE HASH_matlab_inform_create( struct, HASH_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold HASH_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  HASH_pointer - HASH pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( HASH_pointer_type ) :: HASH_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 3
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, HASH_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( HASH_pointer%pointer,              &
        'status', HASH_pointer%status )
      CALL MATLAB_create_integer_component( HASH_pointer%pointer,              &
         'alloc_status', HASH_pointer%alloc_status )
      CALL MATLAB_create_char_component( HASH_pointer%pointer,                 &
        'bad_alloc', HASH_pointer%bad_alloc )

      RETURN

!  End of subroutine HASH_matlab_inform_create

      END SUBROUTINE HASH_matlab_inform_create

!-*-  H A S H _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE HASH_matlab_inform_get( HASH_inform, HASH_pointer )

!  --------------------------------------------------------------

!  Set HASH_inform values from matlab pointers

!  Arguments

!  HASH_inform - HASH inform structure
!  HASH_pointer - HASH pointer structure

!  --------------------------------------------------------------

      TYPE ( HASH_inform_type ) :: HASH_inform
      TYPE ( HASH_pointer_type ) :: HASH_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( HASH_inform%status,                             &
                               mxGetPr( HASH_pointer%status ) )
      CALL MATLAB_copy_to_ptr( HASH_inform%alloc_status,                       &
                               mxGetPr( HASH_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( HASH_pointer%pointer,                           &
                               'bad_alloc', HASH_inform%bad_alloc )

      RETURN

!  End of subroutine HASH_matlab_inform_get

      END SUBROUTINE HASH_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ H A S H _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_HASH_MATLAB_TYPES
