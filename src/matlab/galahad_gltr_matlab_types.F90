#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 16/02/2010 AT 13:30 GMT.

!-*-*-*-  G A L A H A D _ G L T R _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_GLTR_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to GLTR

      USE GALAHAD_MATLAB
      USE GALAHAD_GLTR_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: GLTR_matlab_control_set, GLTR_matlab_control_get,              &
                GLTR_matlab_inform_create, GLTR_matlab_inform_get

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

      TYPE, PUBLIC :: GLTR_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: iter, iter_pass2
        mwPointer :: multiplier, mnormx, piv, curv, rayleigh, leftmost
        mwPointer :: negative_curvature, hard_case
      END TYPE
    CONTAINS

!-*-  G L T R _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE GLTR_matlab_control_set( ps, GLTR_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to GLTR

!  Arguments

!  ps - given pointer to the structure
!  GLTR_control - GLTR control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( GLTR_control_type ) :: GLTR_control

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
                                 pc, GLTR_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, GLTR_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, GLTR_control%print_level )
        CASE( 'itmax' )
          CALL MATLAB_get_value( ps, 'itmax',                                  &
                                 pc, GLTR_control%itmax )
        CASE( 'Lanczos_itmax' )
          CALL MATLAB_get_value( ps, 'Lanczos_itmax',                          &
                                 pc, GLTR_control%Lanczos_itmax )
        CASE( 'extra_vectors' )
          CALL MATLAB_get_value( ps, 'extra_vectors',                          &
                                 pc, GLTR_control%extra_vectors )
        CASE( 'stop_relative' )
          CALL MATLAB_get_value( ps, 'stop_relative',                          &
                                 pc, GLTR_control%stop_relative )
        CASE( 'stop_absolute' )
          CALL MATLAB_get_value( ps, 'stop_absolute',                          &
                                 pc, GLTR_control%stop_absolute )
        CASE( 'fraction_opt' )
          CALL MATLAB_get_value( ps, 'fraction_opt',                           &
                                 pc, GLTR_control%fraction_opt )
        CASE( 'rminvr_zero' )
          CALL MATLAB_get_value( ps, 'rminvr_zero',                            &
                                 pc, GLTR_control%rminvr_zero )
        CASE( 'f_0' )
          CALL MATLAB_get_value( ps, 'f_0',                                    &
                                 pc, GLTR_control%f_0 )
        CASE( 'unitm' )
          CALL MATLAB_get_value( ps, 'unitm',                                  &
                                 pc, GLTR_control%unitm )
        CASE( 'steihaug_toint' )
          CALL MATLAB_get_value( ps, 'steihaug_toint',                         &
                                 pc, GLTR_control%steihaug_toint )
        CASE( 'boundary' )
          CALL MATLAB_get_value( ps, 'boundary',                               &
                                 pc, GLTR_control%boundary )
        CASE( 'equality_problem' )
          CALL MATLAB_get_value( ps, 'equality_problem',                       &
                                 pc, GLTR_control%equality_problem )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, GLTR_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, GLTR_control%deallocate_error_fatal )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, GLTR_control%prefix, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine GLTR_matlab_control_set

      END SUBROUTINE GLTR_matlab_control_set

!-*-  G L T R _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE GLTR_matlab_control_get( struct, GLTR_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to GLTR

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  GLTR_control - GLTR control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( GLTR_control_type ) :: GLTR_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 18
      CHARACTER ( LEN = 22 ), PARAMETER :: finform( ninform ) = (/             &
           'error                 ', 'out                   ',                 &
           'print_level           ', 'itmax                 ',                 &
           'Lanczos_itmax         ', 'extra_vectors         ',                 &
           'stop_relative         ', 'stop_absolute         ',                 &
           'fraction_opt          ', 'rminvr_zero           ',                 &
           'f_0                   ', 'unitm                 ',                 &
           'steihaug_toint        ', 'boundary              ',                 &
           'equality_problem      ', 'space_critical        ',                 &
           'deallocate_error_fatal', 'prefix                '   /)

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
                                  GLTR_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  GLTR_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  GLTR_control%print_level )
      CALL MATLAB_fill_component( pointer, 'itmax',                            &
                                  GLTR_control%itmax )
      CALL MATLAB_fill_component( pointer, 'Lanczos_itmax',                    &
                                  GLTR_control%Lanczos_itmax )
      CALL MATLAB_fill_component( pointer, 'extra_vectors',                    &
                                  GLTR_control%extra_vectors )
      CALL MATLAB_fill_component( pointer, 'stop_relative',                    &
                                  GLTR_control%stop_relative )
      CALL MATLAB_fill_component( pointer, 'stop_absolute',                    &
                                  GLTR_control%stop_absolute )
      CALL MATLAB_fill_component( pointer, 'fraction_opt',                     &
                                  GLTR_control%fraction_opt )
      CALL MATLAB_fill_component( pointer, 'rminvr_zero',                      &
                                  GLTR_control%rminvr_zero )
      CALL MATLAB_fill_component( pointer, 'f_0',                              &
                                  GLTR_control%f_0 )
      CALL MATLAB_fill_component( pointer, 'unitm',                            &
                                  GLTR_control%unitm )
      CALL MATLAB_fill_component( pointer, 'steihaug_toint',                   &
                                  GLTR_control%steihaug_toint )
      CALL MATLAB_fill_component( pointer, 'boundary',                         &
                                  GLTR_control%boundary )
      CALL MATLAB_fill_component( pointer, 'equality_problem',                 &
                                  GLTR_control%equality_problem )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  GLTR_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal ',          &
                                  GLTR_control%deallocate_error_fatal  )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  GLTR_control%prefix )

      RETURN

!  End of subroutine GLTR_matlab_control_get

      END SUBROUTINE GLTR_matlab_control_get

!-*- G L T R _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE GLTR_matlab_inform_create( struct, GLTR_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold GLTR_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  GLTR_pointer - GLTR pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( GLTR_pointer_type ) :: GLTR_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 13
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'bad_alloc            ', 'iter                 ',                   &
           'iter_pass2           ', 'multiplier           ',                   &
           'mnormx               ', 'piv                  ',                   &
           'curv                 ', 'rayleigh             ',                   &
           'leftmost             ', 'negative_curvature   ',                   &
           'hard_case            '                              /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, GLTR_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        GLTR_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( GLTR_pointer%pointer,              &
        'status', GLTR_pointer%status )
      CALL MATLAB_create_integer_component( GLTR_pointer%pointer,              &
         'alloc_status', GLTR_pointer%alloc_status )
      CALL MATLAB_create_char_component( GLTR_pointer%pointer,                 &
        'bad_alloc', GLTR_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( GLTR_pointer%pointer,              &
         'iter', GLTR_pointer%iter )
      CALL MATLAB_create_integer_component( GLTR_pointer%pointer,              &
        'iter_pass2', GLTR_pointer%iter_pass2 )
      CALL MATLAB_create_logical_component( GLTR_pointer%pointer,              &
        'negative_curvature', GLTR_pointer%negative_curvature )
      CALL MATLAB_create_logical_component( GLTR_pointer%pointer,              &
        'hard_case', GLTR_pointer%hard_case )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'multiplier',  GLTR_pointer%multiplier )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'mnormx',  GLTR_pointer%mnormx )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'piv',  GLTR_pointer%piv )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'curv',  GLTR_pointer%curv )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'leftmost',  GLTR_pointer%leftmost )
      CALL MATLAB_create_real_component( GLTR_pointer%pointer,                 &
        'rayleigh', GLTR_pointer%rayleigh )

      RETURN

!  End of subroutine GLTR_matlab_inform_create

      END SUBROUTINE GLTR_matlab_inform_create

!-*-  G L T R _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-

      SUBROUTINE GLTR_matlab_inform_get( GLTR_inform, GLTR_pointer )

!  --------------------------------------------------------------

!  Set GLTR_inform values from matlab pointers

!  Arguments

!  GLTR_inform - GLTR inform structure
!  GLTR_pointer - GLTR pointer structure

!  --------------------------------------------------------------

      TYPE ( GLTR_info_type ) :: GLTR_inform
      TYPE ( GLTR_pointer_type ) :: GLTR_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( GLTR_inform%status,                             &
                               mxGetPr( GLTR_pointer%status ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%alloc_status,                       &
                               mxGetPr( GLTR_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( GLTR_pointer%pointer,                           &
                               'bad_alloc', GLTR_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( GLTR_inform%iter,                               &
                               mxGetPr( GLTR_pointer%iter ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%iter_pass2,                         &
                               mxGetPr( GLTR_pointer%iter_pass2 ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%multiplier,                         &
                               mxGetPr( GLTR_pointer%multiplier ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%mnormx,                             &
                               mxGetPr( GLTR_pointer%mnormx ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%piv,                                &
                               mxGetPr( GLTR_pointer%piv ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%curv,                               &
                               mxGetPr( GLTR_pointer%curv ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%rayleigh,                           &
                               mxGetPr( GLTR_pointer%rayleigh ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%leftmost,                           &
                               mxGetPr( GLTR_pointer%leftmost ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%negative_curvature,                 &
                               mxGetPr( GLTR_pointer%negative_curvature ) )
      CALL MATLAB_copy_to_ptr( GLTR_inform%hard_case,                          &
                               mxGetPr( GLTR_pointer%hard_case ) )

      RETURN

!  End of subroutine GLTR_matlab_inform_get

      END SUBROUTINE GLTR_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ G L T R _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_GLTR_MATLAB_TYPES

