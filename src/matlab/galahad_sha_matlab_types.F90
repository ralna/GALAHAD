#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.0 - 02/03/2017 AT 10:00 GMT.

!-**-*-*-  G A L A H A D _ S H A _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 2nd, 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SHA_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SHA

      USE GALAHAD_MATLAB
      USE GALAHAD_SHA_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SHA_matlab_control_set, SHA_matlab_control_get,                &
                SHA_matlab_inform_create, SHA_matlab_inform_get

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

      TYPE, PUBLIC :: SHA_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: max_degree, differences_needed, max_reduced_degree
      END TYPE

    CONTAINS

!-*-*-  S H A _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E  -*-*-

      SUBROUTINE SHA_matlab_control_set( ps, SHA_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SHA

!  Arguments

!  ps - given pointer to the structure
!  SHA_control - SHA control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( SHA_control_type ) :: SHA_control

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
                                 pc, SHA_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, SHA_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, SHA_control%print_level )
        CASE( 'approximation_algorithm' )
          CALL MATLAB_get_value( ps, 'approximation_algorithm',                &
                                 pc, SHA_control%approximation_algorithm )
        CASE( 'dense_linear_solver' )
          CALL MATLAB_get_value( ps, 'dense_linear_solver',                    &
                                 pc, SHA_control%dense_linear_solver )
        CASE( 'max_sparse_degree' )
          CALL MATLAB_get_value( ps, 'max_sparse_degree',                      &
                                 pc, SHA_control%max_sparse_degree )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, SHA_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, SHA_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, SHA_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine SHA_matlab_control_set

      END SUBROUTINE SHA_matlab_control_set

!-*-*-  S H A _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE SHA_matlab_control_get( struct, SHA_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SHA

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SHA_control - SHA control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SHA_control_type ) :: SHA_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'approximation_algorithm        ', &
         'dense_linear_solver            ', 'max_sparse_degree              ', &
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
                                  SHA_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  SHA_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  SHA_control%print_level )
      CALL MATLAB_fill_component( pointer, 'approximation_algorithm',          &
                                  SHA_control%approximation_algorithm )
      CALL MATLAB_fill_component( pointer, 'dense_linear_solver',              &
                                  SHA_control%dense_linear_solver )
      CALL MATLAB_fill_component( pointer, 'max_sparse_degree',                &
                                  SHA_control%max_sparse_degree )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  SHA_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  SHA_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  SHA_control%prefix )

      RETURN

!  End of subroutine SHA_matlab_control_get

      END SUBROUTINE SHA_matlab_control_get

!-*-  S H A _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E  -*-

      SUBROUTINE SHA_matlab_inform_create( struct, SHA_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SHA_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SHA_pointer - SHA pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SHA_pointer_type ) :: SHA_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'max_degree           ', 'differences_needed   ',                   &
           'max_reduced_degree   ', 'bad_alloc            '      /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SHA_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( SHA_pointer%pointer,               &
        'status', SHA_pointer%status )
      CALL MATLAB_create_integer_component( SHA_pointer%pointer,               &
         'alloc_status', SHA_pointer%alloc_status )
      CALL MATLAB_create_integer_component( SHA_pointer%pointer,               &
         'max_degree', SHA_pointer%max_degree )
      CALL MATLAB_create_integer_component( SHA_pointer%pointer,               &
         'differences_needed', SHA_pointer%differences_needed )
      CALL MATLAB_create_integer_component( SHA_pointer%pointer,               &
         'max_reduced_degree', SHA_pointer%max_reduced_degree )
      CALL MATLAB_create_char_component( SHA_pointer%pointer,                  &
        'bad_alloc', SHA_pointer%bad_alloc )

      RETURN

!  End of subroutine SHA_matlab_inform_create

      END SUBROUTINE SHA_matlab_inform_create

!-*-*-  S H A _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE SHA_matlab_inform_get( SHA_inform, SHA_pointer )

!  --------------------------------------------------------------

!  Set SHA_inform values from matlab pointers

!  Arguments

!  SHA_inform - SHA inform structure
!  SHA_pointer - SHA pointer structure

!  --------------------------------------------------------------

      TYPE ( SHA_inform_type ) :: SHA_inform
      TYPE ( SHA_pointer_type ) :: SHA_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SHA_inform%status,                              &
                               mxGetPr( SHA_pointer%status ) )
      CALL MATLAB_copy_to_ptr( SHA_inform%alloc_status,                        &
                               mxGetPr( SHA_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( SHA_inform%max_degree,                          &
                               mxGetPr( SHA_pointer%max_degree ) )
      CALL MATLAB_copy_to_ptr( SHA_inform%differences_needed,                  &
                               mxGetPr( SHA_pointer%differences_needed ) )
      CALL MATLAB_copy_to_ptr( SHA_inform%max_reduced_degree,                  &
                               mxGetPr( SHA_pointer%max_reduced_degree ) )
      CALL MATLAB_copy_to_ptr( SHA_pointer%pointer,                            &
                               'bad_alloc', SHA_inform%bad_alloc )

      RETURN

!  End of subroutine SHA_matlab_inform_get

      END SUBROUTINE SHA_matlab_inform_get

!-*-*-*-*-  E N D  o f  G A L A H A D _ S H A _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SHA_MATLAB_TYPES

