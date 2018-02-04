#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 08/03/2010 AT 10:30 GMT.

!-**-*-*-  G A L A H A D _ I R _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 10th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_IR_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to IR

      USE GALAHAD_MATLAB
      USE GALAHAD_IR_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: IR_matlab_control_set, IR_matlab_control_get,                  &
                IR_matlab_inform_create, IR_matlab_inform_get

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

      TYPE, PUBLIC :: IR_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
      END TYPE 

    CONTAINS

!-*-*-  I R _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E   -*-*-

      SUBROUTINE IR_matlab_control_set( ps, IR_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to IR

!  Arguments

!  ps - given pointer to the structure
!  IR_control - IR control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( IR_control_type ) :: IR_control

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
                                 pc, IR_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, IR_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, IR_control%print_level )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, IR_control%itref_max )
        CASE( 'acceptable_residual_relative' )
          CALL MATLAB_get_value( ps, 'acceptable_residual_relative',           &
                                 pc, IR_control%acceptable_residual_relative )
        CASE( 'acceptable_residual_absolute' )
          CALL MATLAB_get_value( ps, 'acceptable_residual_absolute',           &
                                 pc, IR_control%acceptable_residual_absolute )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, IR_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, IR_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, IR_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine IR_matlab_control_set

      END SUBROUTINE IR_matlab_control_set

!-*-*-  I R _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE IR_matlab_control_get( struct, IR_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to IR

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  IR_control - IR control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( IR_control_type ) :: IR_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'itref_max                      ', &
         'acceptable_residual_relative   ', 'acceptable_residual_absolute   ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'prefix                         '                      /)

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
                                  IR_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  IR_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  IR_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  IR_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_relative',     &
                                  IR_control%acceptable_residual_relative )
      CALL MATLAB_fill_component( pointer, 'acceptable_residual_absolute',     &
                                  IR_control%acceptable_residual_absolute )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  IR_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  IR_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  IR_control%prefix )

      RETURN

!  End of subroutine IR_matlab_control_get

      END SUBROUTINE IR_matlab_control_get

!-*-  I R _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E   -*-

      SUBROUTINE IR_matlab_inform_create( struct, IR_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold IR_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  IR_pointer - IR pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( IR_pointer_type ) :: IR_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 3
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, IR_pointer%pointer,     &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( IR_pointer%pointer,                &
        'status', IR_pointer%status )
      CALL MATLAB_create_integer_component( IR_pointer%pointer,                &
         'alloc_status', IR_pointer%alloc_status )
      CALL MATLAB_create_char_component( IR_pointer%pointer,                   &
        'bad_alloc', IR_pointer%bad_alloc )

      RETURN

!  End of subroutine IR_matlab_inform_create

      END SUBROUTINE IR_matlab_inform_create

!-*-*-  I R _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE IR_matlab_inform_get( IR_inform, IR_pointer )

!  --------------------------------------------------------------

!  Set IR_inform values from matlab pointers

!  Arguments

!  IR_inform - IR inform structure
!  IR_pointer - IR pointer structure

!  --------------------------------------------------------------

      TYPE ( IR_inform_type ) :: IR_inform
      TYPE ( IR_pointer_type ) :: IR_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( IR_inform%status,                               &
                               mxGetPr( IR_pointer%status ) )
      CALL MATLAB_copy_to_ptr( IR_inform%alloc_status,                         &
                               mxGetPr( IR_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( IR_pointer%pointer,                             &
                               'bad_alloc', IR_inform%bad_alloc )

      RETURN

!  End of subroutine IR_matlab_inform_get

      END SUBROUTINE IR_matlab_inform_get

!-*-*-*-*-  E N D  o f  G A L A H A D _ I R _ T Y P E S   M O D U L E  -*-*-*-*-

    END MODULE GALAHAD_IR_MATLAB_TYPES

