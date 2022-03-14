#include <fintrf.h>

!  THIS VERSION: GALAHAD 4.0 - 2022-03-14 AT 09:30 GMT.

!-**-*-*-  G A L A H A D _ L H S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 14th, 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LHS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LHS

      USE GALAHAD_MATLAB
      USE GALAHAD_LHS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LHS_matlab_control_set, LHS_matlab_control_get,                &
                LHS_matlab_inform_create, LHS_matlab_inform_get

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

      TYPE, PUBLIC :: LHS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
      END TYPE

    CONTAINS

!-*-*-  L H S _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E  -*-*-

      SUBROUTINE LHS_matlab_control_set( ps, LHS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LHS

!  Arguments

!  ps - given pointer to the structure
!  LHS_control - LHS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LHS_control_type ) :: LHS_control

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
                                 pc, LHS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LHS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LHS_control%print_level )
        CASE( 'duplication' )
          CALL MATLAB_get_value( ps, 'duplication',                            &
                                 pc, LHS_control%duplication )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LHS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LHS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LHS_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LHS_matlab_control_set

      END SUBROUTINE LHS_matlab_control_set

!-*-*-  L H S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE LHS_matlab_control_get( struct, LHS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LHS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LHS_control - LHS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LHS_control_type ) :: LHS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 7
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'duplication                    ', &
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
                                  LHS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LHS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LHS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'duplication',                      &
                                  LHS_control%duplication )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LHS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LHS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LHS_control%prefix )

      RETURN

!  End of subroutine LHS_matlab_control_get

      END SUBROUTINE LHS_matlab_control_get

!-*-  L H S _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E  -*-

      SUBROUTINE LHS_matlab_inform_create( struct, LHS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LHS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LHS_pointer - LHS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LHS_pointer_type ) :: LHS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 3
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LHS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( LHS_pointer%pointer,               &
        'status', LHS_pointer%status )
      CALL MATLAB_create_integer_component( LHS_pointer%pointer,               &
         'alloc_status', LHS_pointer%alloc_status )
      CALL MATLAB_create_char_component( LHS_pointer%pointer,                  &
        'bad_alloc', LHS_pointer%bad_alloc )

      RETURN

!  End of subroutine LHS_matlab_inform_create

      END SUBROUTINE LHS_matlab_inform_create

!-*-*-  L H S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE LHS_matlab_inform_get( LHS_inform, LHS_pointer )

!  --------------------------------------------------------------

!  Set LHS_inform values from matlab pointers

!  Arguments

!  LHS_inform - LHS inform structure
!  LHS_pointer - LHS pointer structure

!  --------------------------------------------------------------

      TYPE ( LHS_inform_type ) :: LHS_inform
      TYPE ( LHS_pointer_type ) :: LHS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LHS_inform%status,                              &
                               mxGetPr( LHS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LHS_inform%alloc_status,                        &
                               mxGetPr( LHS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LHS_pointer%pointer,                            &
                               'bad_alloc', LHS_inform%bad_alloc )

      RETURN

!  End of subroutine LHS_matlab_inform_get

      END SUBROUTINE LHS_matlab_inform_get

!-*-*-*-*-  E N D  o f  G A L A H A D _ L H S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LHS_MATLAB_TYPES
