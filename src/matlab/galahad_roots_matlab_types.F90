#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.2 - 06/03/2019 AT 09:17 GMT.

!-*-*-  G A L A H A D _ R O O T S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.2. March 6th, 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_ROOTS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to ROOTS

      USE GALAHAD_MATLAB
      USE GALAHAD_ROOTS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: ROOTS_matlab_control_set, ROOTS_matlab_control_get,            &
                ROOTS_matlab_inform_create, ROOTS_matlab_inform_get

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

      TYPE, PUBLIC :: ROOTS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
      END TYPE
    CONTAINS

!-*-  R O O T S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE ROOTS_matlab_control_set( ps, ROOTS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to ROOTS

!  Arguments

!  ps - given pointer to the structure
!  ROOTS_control - ROOTS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( ROOTS_control_type ) :: ROOTS_control

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
                                 pc, ROOTS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, ROOTS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, ROOTS_control%print_level )
        CASE( 'tol' )
          CALL MATLAB_get_value( ps, 'tol',                                    &
                                 pc, ROOTS_control%tol )
        CASE( 'zero_coef' )
          CALL MATLAB_get_value( ps, 'zero_coef',                              &
                                 pc, ROOTS_control%zero_coef )
        CASE( 'zero_f' )
          CALL MATLAB_get_value( ps, 'zero_f',                          &
                                 pc, ROOTS_control%zero_f )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, ROOTS_control%space_critical )
        CASE( 'deallocate_error_fatal ' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal ',                &
                                 pc, ROOTS_control%deallocate_error_fatal  )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, ROOTS_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine ROOTS_matlab_control_set

      END SUBROUTINE ROOTS_matlab_control_set

!-*-  R O O T S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE ROOTS_matlab_control_get( struct, ROOTS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to ROOTS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  ROOTS_control - ROOTS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ROOTS_control_type ) :: ROOTS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 25 ), PARAMETER :: finform( ninform ) = (/             &
           'error                    ', 'out                      ',           &
           'print_level              ', 'tol                      ',           &
           'zero_coef                ', 'zero_f                   ',           &
           'space_critical           ', 'deallocate_error_fatal   ',           &
           'prefix                   ' /)

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
                                  ROOTS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  ROOTS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  ROOTS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'tol',                              &
                                  ROOTS_control%tol )
      CALL MATLAB_fill_component( pointer, 'zero_coef',                        &
                                  ROOTS_control%zero_coef )
      CALL MATLAB_fill_component( pointer, 'zero_f',                           &
                                  ROOTS_control%zero_f )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  ROOTS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  ROOTS_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  ROOTS_control%prefix )

      RETURN

!  End of subroutine ROOTS_matlab_control_get

      END SUBROUTINE ROOTS_matlab_control_get

!-*- R O O T S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -

      SUBROUTINE ROOTS_matlab_inform_create( struct, ROOTS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold ROOTS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  ROOTS_pointer - ROOTS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( ROOTS_pointer_type ) :: ROOTS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 3
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, ROOTS_pointer%pointer,  &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        ROOTS_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( ROOTS_pointer%pointer,             &
        'status', ROOTS_pointer%status )
      CALL MATLAB_create_integer_component( ROOTS_pointer%pointer,             &
         'alloc_status', ROOTS_pointer%alloc_status )
      CALL MATLAB_create_char_component( ROOTS_pointer%pointer,                &
        'bad_alloc', ROOTS_pointer%bad_alloc )

      RETURN

!  End of subroutine ROOTS_matlab_inform_create

      END SUBROUTINE ROOTS_matlab_inform_create

!-*-*  R O O T S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   *-

      SUBROUTINE ROOTS_matlab_inform_get( ROOTS_inform, ROOTS_pointer )

!  --------------------------------------------------------------

!  Set ROOTS_inform values from matlab pointers

!  Arguments

!  ROOTS_inform - ROOTS inform structure
!  ROOTS_pointer - ROOTS pointer structure

!  --------------------------------------------------------------

      TYPE ( ROOTS_inform_type ) :: ROOTS_inform
      TYPE ( ROOTS_pointer_type ) :: ROOTS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( ROOTS_inform%status,                            &
                               mxGetPr( ROOTS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( ROOTS_inform%alloc_status,                      &
                               mxGetPr( ROOTS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( ROOTS_pointer%pointer,                          &
                               'bad_alloc', ROOTS_inform%bad_alloc )

      RETURN

!  End of subroutine ROOTS_matlab_inform_get

      END SUBROUTINE ROOTS_matlab_inform_get

!-*-*-  E N D  o f  G A L A H A D _ R O O T S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_ROOTS_MATLAB_TYPES
