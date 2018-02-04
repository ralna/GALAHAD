#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.0 - 02/03/2017 AT 10:00 GMT.

!-**-*-*-  G A L A H A D _ L M S _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.0. March 2nd, 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LMS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to LMS

      USE GALAHAD_MATLAB
      USE GALAHAD_LMS_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LMS_matlab_control_set, LMS_matlab_control_get,                &
                LMS_matlab_inform_create, LMS_matlab_inform_get

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

      TYPE, PUBLIC :: LMS_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, setup, form, apply
        mwPointer :: clock_total, clock_setup, clock_form, clock_apply
      END TYPE

      TYPE, PUBLIC :: LMS_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, length, updates_skipped, bad_alloc
        TYPE ( LMS_time_pointer_type ) :: time_pointer
      END TYPE

    CONTAINS

!-*-*-  L M S _ M A T L A B _ C O N T R O L _ S E T   S U B R O U T I N E  -*-*-

      SUBROUTINE LMS_matlab_control_set( ps, LMS_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to LMS

!  Arguments

!  ps - given pointer to the structure
!  LMS_control - LMS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( LMS_control_type ) :: LMS_control

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
                                 pc, LMS_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, LMS_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, LMS_control%print_level )
        CASE( 'memory_length' )
          CALL MATLAB_get_value( ps, 'memory_length',                          &
                                 pc, LMS_control%memory_length )
        CASE( 'method' )
          CALL MATLAB_get_value( ps, 'method',                                 &
                                 pc, LMS_control%method )
        CASE( 'any_method' )
          CALL MATLAB_get_value( ps, 'any_method',                             &
                                 pc, LMS_control%any_method )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, LMS_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, LMS_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, LMS_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine LMS_matlab_control_set

      END SUBROUTINE LMS_matlab_control_set

!-*-*-  L M S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-*-

      SUBROUTINE LMS_matlab_control_get( struct, LMS_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to LMS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LMS_control - LMS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LMS_control_type ) :: LMS_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 9
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'memory_length                  ', &
         'method                         ', 'any_method                     ', &
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
                                  LMS_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  LMS_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  LMS_control%print_level )
      CALL MATLAB_fill_component( pointer, 'memory_length',                    &
                                  LMS_control%memory_length )
      CALL MATLAB_fill_component( pointer, 'method',                           &
                                  LMS_control%method )
      CALL MATLAB_fill_component( pointer, 'any_method',                       &
                                  LMS_control%any_method )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  LMS_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  LMS_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  LMS_control%prefix )

      RETURN

!  End of subroutine LMS_matlab_control_get

      END SUBROUTINE LMS_matlab_control_get

!-*-  L M S _ M A T L A B _ I N F O R M _ C R E A T E   S U B R O U T I N E  -*-

      SUBROUTINE LMS_matlab_inform_create( struct, LMS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold LMS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  LMS_pointer - LMS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( LMS_pointer_type ) :: LMS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 6
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'length               ', 'updates_skipped      ',                   &
           'bad_alloc            ', 'time                 ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'setup                ',                   &
           'form                 ', 'apply                ',                   &
           'clock_total          ', 'clock_setup          ',                   &
           'clock_form           ', 'clock_apply          '          /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, LMS_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( LMS_pointer%pointer,               &
        'status', LMS_pointer%status )
      CALL MATLAB_create_integer_component( LMS_pointer%pointer,               &
         'alloc_status', LMS_pointer%alloc_status )
      CALL MATLAB_create_integer_component( LMS_pointer%pointer,               &
         'length', LMS_pointer%length )
      CALL MATLAB_create_logical_component( LMS_pointer%pointer,               &
         'updates_skipped', LMS_pointer%updates_skipped )
      CALL MATLAB_create_char_component( LMS_pointer%pointer,                  &
        'bad_alloc', LMS_pointer%bad_alloc )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( LMS_pointer%pointer,                    &
        'time', LMS_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'total', LMS_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'setup', LMS_pointer%time_pointer%setup )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'form', LMS_pointer%time_pointer%form )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'apply', LMS_pointer%time_pointer%apply )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'clock_total', LMS_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'clock_setup', LMS_pointer%time_pointer%clock_setup )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'clock_form', LMS_pointer%time_pointer%clock_form )
      CALL MATLAB_create_real_component( LMS_pointer%time_pointer%pointer,     &
        'clock_apply', LMS_pointer%time_pointer%clock_apply )

      RETURN

!  End of subroutine LMS_matlab_inform_create

      END SUBROUTINE LMS_matlab_inform_create

!-*-*-  L M S _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE LMS_matlab_inform_get( LMS_inform, LMS_pointer )

!  --------------------------------------------------------------

!  Set LMS_inform values from matlab pointers

!  Arguments

!  LMS_inform - LMS inform structure
!  LMS_pointer - LMS pointer structure

!  --------------------------------------------------------------

      TYPE ( LMS_inform_type ) :: LMS_inform
      TYPE ( LMS_pointer_type ) :: LMS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( LMS_inform%status,                              &
                               mxGetPr( LMS_pointer%status ) )
      CALL MATLAB_copy_to_ptr( LMS_inform%alloc_status,                        &
                               mxGetPr( LMS_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( LMS_inform%length,                              &
                               mxGetPr( LMS_pointer%length ) )
      CALL MATLAB_copy_to_ptr( LMS_inform%updates_skipped,                     &
                               mxGetPr( LMS_pointer%updates_skipped ) )
      CALL MATLAB_copy_to_ptr( LMS_pointer%pointer,                            &
                               'bad_alloc', LMS_inform%bad_alloc )

!  time components

      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%total, wp ),              &
                               mxGetPr( LMS_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%setup, wp ),              &
                               mxGetPr( LMS_pointer%time_pointer%setup ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%form, wp ),               &
                               mxGetPr( LMS_pointer%time_pointer%form ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%apply, wp ),              &
                               mxGetPr( LMS_pointer%time_pointer%apply ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%clock_total, wp ),        &
                               mxGetPr( LMS_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%clock_setup, wp ),        &
                               mxGetPr( LMS_pointer%time_pointer%clock_setup ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%clock_form, wp ),         &
                               mxGetPr( LMS_pointer%time_pointer%clock_form ) )
      CALL MATLAB_copy_to_ptr( REAL( LMS_inform%time%clock_apply, wp ),        &
                               mxGetPr( LMS_pointer%time_pointer%clock_apply ) )

      RETURN

!  End of subroutine LMS_matlab_inform_get

      END SUBROUTINE LMS_matlab_inform_get

!-*-*-*-*-  E N D  o f  G A L A H A D _ L M S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_LMS_MATLAB_TYPES

