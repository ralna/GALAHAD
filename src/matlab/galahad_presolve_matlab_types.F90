#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 01/02/2011 AT 19:00 GMT.

!-   G A L A H A D _ P R E S O L V E _ M A T L A B _ T Y P E S   M O D U L E  -

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 1st, 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_PRESOLVE_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to PRESOLVE

      USE GALAHAD_MATLAB
      USE GALAHAD_PRESOLVE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: PRESOLVE_matlab_control_set, PRESOLVE_matlab_control_get,      &
                PRESOLVE_matlab_inform_create, PRESOLVE_matlab_inform_get

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

      TYPE, PUBLIC :: PRESOLVE_pointer_type
        mwPointer :: pointer
        mwPointer :: status, nbr_transforms, message
      END TYPE 
    CONTAINS

!- P R E S O L V E _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E  -

      SUBROUTINE PRESOLVE_matlab_control_set( ps, PRESOLVE_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QP

!  Arguments

!  ps - given pointer to the structure
!  PRESOLVE_control - PRESOLVE control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'termination' )
          CALL MATLAB_get_value( ps, 'termination',                           &
                                 pc, PRESOLVE_control%termination )
        CASE( 'max_nbr_transforms' )
          CALL MATLAB_get_value( ps, 'max_nbr_transforms',                     &
                                 pc, PRESOLVE_control%max_nbr_transforms )
        CASE( 'max_nbr_passes' )
          CALL MATLAB_get_value( ps, 'max_nbr_passes',                         &
                                 pc, PRESOLVE_control%max_nbr_passes )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, PRESOLVE_control%out )
        CASE( 'errout' )
          CALL MATLAB_get_value( ps, 'errout',                                 &
                                 pc, PRESOLVE_control%errout )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, PRESOLVE_control%print_level )
        CASE( 'primal_constraints_freq' )
          CALL MATLAB_get_value( ps, 'primal_constraints_freq',                &
                                 pc, PRESOLVE_control%primal_constraints_freq )
        CASE( 'dual_constraints_freq' )
          CALL MATLAB_get_value( ps, 'dual_constraints_freq',                  &
                                 pc, PRESOLVE_control%dual_constraints_freq )
        CASE( 'singleton_columns_freq' )
          CALL MATLAB_get_value( ps, 'singleton_columns_freq',                 &
                                 pc, PRESOLVE_control%singleton_columns_freq )
        CASE( 'doubleton_columns_freq' )
          CALL MATLAB_get_value( ps, 'doubleton_columns_freq',                 &
                                 pc, PRESOLVE_control%doubleton_columns_freq )
        CASE( 'unc_variables_freq' )
          CALL MATLAB_get_value( ps, 'unc_variables_freq',                     &
                                 pc, PRESOLVE_control%unc_variables_freq )
        CASE( 'dependent_variables_freq' )
          CALL MATLAB_get_value( ps, 'dependent_variables_freq', pc,           &
                                 PRESOLVE_control%dependent_variables_freq )
        CASE( 'sparsify_rows_freq' )
          CALL MATLAB_get_value( ps, 'sparsify_rows_freq',                     &
                                 pc, PRESOLVE_control%sparsify_rows_freq )
        CASE( 'max_fill' )
          CALL MATLAB_get_value( ps, 'max_fill',                               &
                                 pc, PRESOLVE_control%max_fill )
        CASE( 'transf_file_nbr' )
          CALL MATLAB_get_value( ps, 'transf_file_nbr',                        &
                                 pc, PRESOLVE_control%transf_file_nbr )
        CASE( 'transf_buffer_size' )
          CALL MATLAB_get_value( ps, 'transf_buffer_size',                     &
                                 pc, PRESOLVE_control%transf_buffer_size )
        CASE( 'transf_file_status' )
          CALL MATLAB_get_value( ps, 'transf_file_status',                     &
                                 pc, PRESOLVE_control%transf_file_status )
        CASE( 'y_sign' )
          CALL MATLAB_get_value( ps, 'y_sign',                                 &
                                 pc, PRESOLVE_control%y_sign )
        CASE( 'inactive_y' )
          CALL MATLAB_get_value( ps, 'inactive_y',                             &
                                 pc, PRESOLVE_control%inactive_y )
        CASE( 'z_sign' )
          CALL MATLAB_get_value( ps, 'z_sign',                                 &
                                 pc, PRESOLVE_control%z_sign )
        CASE( 'inactive_z' )
          CALL MATLAB_get_value( ps, 'inactive_z',                             &
                                 pc, PRESOLVE_control%inactive_z )
        CASE( 'final_x_bounds' )
          CALL MATLAB_get_value( ps, 'final_x_bounds',                         &
                                 pc, PRESOLVE_control%final_x_bounds )
        CASE( 'final_z_bounds' )
          CALL MATLAB_get_value( ps, 'final_z_bounds',                         &
                                 pc, PRESOLVE_control%final_z_bounds )
        CASE( 'final_c_bounds' )
          CALL MATLAB_get_value( ps, 'final_c_bounds',                         &
                                 pc, PRESOLVE_control%final_c_bounds )
        CASE( 'final_y_bounds' )
          CALL MATLAB_get_value( ps, 'final_y_bounds',                         &
                                 pc, PRESOLVE_control%final_y_bounds )
        CASE( 'check_primal_feasibility' )
          CALL MATLAB_get_value( ps, 'check_primal_feasibility', pc,           &
                                 PRESOLVE_control%check_primal_feasibility )
        CASE( 'check_dual_feasibility' )
          CALL MATLAB_get_value( ps, 'check_dual_feasibility',                 &
                                 pc, PRESOLVE_control%check_dual_feasibility )
        CASE( 'c_accuracy' )
          CALL MATLAB_get_value( ps, 'c_accuracy',                             &
                                 pc, PRESOLVE_control%c_accuracy )
        CASE( 'z_accuracy' )
          CALL MATLAB_get_value( ps, 'z_accuracy',                             &
                                 pc, PRESOLVE_control%z_accuracy )
        CASE( 'infinity' )
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, PRESOLVE_control%infinity )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, PRESOLVE_control%pivot_tol )
        CASE( 'min_rel_improve' )
          CALL MATLAB_get_value( ps, 'min_rel_improve',                        &
                                 pc, PRESOLVE_control%min_rel_improve )
        CASE( 'max_growth_factor' )
          CALL MATLAB_get_value( ps, 'max_growth_factor',                      &
                                 pc, PRESOLVE_control%max_growth_factor )
        CASE( 'dual_transformations' )
          CALL MATLAB_get_value( ps, 'dual_transformations',                   &
                                 pc, PRESOLVE_control%dual_transformations )
        CASE( 'redundant_xc' )
          CALL MATLAB_get_value( ps, 'redundant_xc',                           &
                                 pc, PRESOLVE_control%redundant_xc )
        CASE( 'get_q' )
          CALL MATLAB_get_value( ps, 'get_q',                                  &
                                 pc, PRESOLVE_control%get_q )
        CASE( 'get_f' )
          CALL MATLAB_get_value( ps, 'get_f',                                  &
                                 pc, PRESOLVE_control%get_f )
        CASE( 'get_g' )
          CALL MATLAB_get_value( ps, 'get_g',                                  &
                                 pc, PRESOLVE_control%get_g )
        CASE( 'get_H' )
          CALL MATLAB_get_value( ps, 'get_H',                                  &
                                 pc, PRESOLVE_control%get_H )
        CASE( 'get_A' )
          CALL MATLAB_get_value( ps, 'get_A',                                  &
                                 pc, PRESOLVE_control%get_A )
        CASE( 'get_x' )
          CALL MATLAB_get_value( ps, 'get_x',                                  &
                                 pc, PRESOLVE_control%get_x )
        CASE( 'get_x_bounds' )
          CALL MATLAB_get_value( ps, 'get_x_bounds',                           &
                                 pc, PRESOLVE_control%get_x_bounds )
        CASE( 'get_z' )
          CALL MATLAB_get_value( ps, 'get_z',                                  &
                                 pc, PRESOLVE_control%get_z )
        CASE( 'get_z_bounds' )
          CALL MATLAB_get_value( ps, 'get_z_bounds',                           &
                                 pc, PRESOLVE_control%get_z_bounds )
        CASE( 'get_c' )
          CALL MATLAB_get_value( ps, 'get_c',                                  &
                                 pc, PRESOLVE_control%get_c )
        CASE( 'get_c_bounds' )
          CALL MATLAB_get_value( ps, 'get_c_bounds',                           &
                                 pc, PRESOLVE_control%get_c_bounds )
        CASE( 'get_y' )
          CALL MATLAB_get_value( ps, 'get_y',                                  &
                                 pc, PRESOLVE_control%get_y )
        CASE( 'get_y_bounds' )
          CALL MATLAB_get_value( ps, 'get_y_bounds',                           &
                                 pc, PRESOLVE_control%get_y_bounds )
        CASE( 'transf_file_name' )
          CALL galmxGetCharacter( ps, 'transf_file_name',                      &
                                  pc, PRESOLVE_control%transf_file_name, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine PRESOLVE_matlab_control_set

      END SUBROUTINE PRESOLVE_matlab_control_set

!-  P R E S O L V E _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E -

      SUBROUTINE PRESOLVE_matlab_control_get( struct, PRESOLVE_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  PRESOLVE_control - PRESOLVE control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 49
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'termination                    ', 'max_nbr_transforms             ', &
         'max_nbr_passes                 ', 'out                            ', &
         'errout                         ', 'print_level                    ', &
         'primal_constraints_freq        ', 'dual_constraints_freq          ', &
         'singleton_columns_freq         ', 'doubleton_columns_freq         ', &
         'unc_variables_freq             ', 'dependent_variables_freq       ', &
         'sparsify_rows_freq             ', 'max_fill                       ', &
         'transf_file_nbr                ', 'transf_buffer_size             ', &
         'transf_file_status             ', 'y_sign                         ', &
         'inactive_y                     ', 'z_sign                         ', &
         'inactive_z                     ', 'final_x_bounds                 ', &
         'final_z_bounds                 ', 'final_c_bounds                 ', &
         'final_y_bounds                 ', 'check_primal_feasibility       ', &
         'check_dual_feasibility         ', 'c_accuracy                     ', &
         'z_accuracy                     ', 'infinity                       ', &
         'pivot_tol                      ', 'min_rel_improve                ', &
         'max_growth_factor              ', 'dual_transformations           ', &
         'redundant_xc                   ', 'get_q                          ', &
         'get_f                          ', 'get_g                          ', &
         'get_H                          ', 'get_A                          ', &
         'get_x                          ', 'get_x_bounds                   ', &
         'get_z                          ', 'get_z_bounds                   ', &
         'get_c                          ', 'get_c_bounds                   ', &
         'get_y                          ', 'get_y_bounds                   ', &
         'transf_file_name               ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, pointer,                &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        pointer = struct
      END IF

!  create the components and get the values

      CALL MATLAB_fill_component( pointer, 'termination',                      &
                                  PRESOLVE_control%termination )
      CALL MATLAB_fill_component( pointer, 'max_nbr_transforms',               &
                                  PRESOLVE_control%max_nbr_transforms )
      CALL MATLAB_fill_component( pointer, 'max_nbr_passes',                   &
                                  PRESOLVE_control%max_nbr_passes )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  PRESOLVE_control%out )
      CALL MATLAB_fill_component( pointer, 'errout',                           &
                                  PRESOLVE_control%errout )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  PRESOLVE_control%print_level )
      CALL MATLAB_fill_component( pointer, 'primal_constraints_freq',          &
                                  PRESOLVE_control%primal_constraints_freq )
      CALL MATLAB_fill_component( pointer, 'dual_constraints_freq',            &
                                  PRESOLVE_control%dual_constraints_freq )
      CALL MATLAB_fill_component( pointer, 'singleton_columns_freq',           &
                                  PRESOLVE_control%singleton_columns_freq )
      CALL MATLAB_fill_component( pointer, 'doubleton_columns_freq',           &
                                  PRESOLVE_control%doubleton_columns_freq )
      CALL MATLAB_fill_component( pointer, 'unc_variables_freq',               &
                                  PRESOLVE_control%unc_variables_freq )
      CALL MATLAB_fill_component( pointer, 'dependent_variables_freq',         &
                                  PRESOLVE_control%dependent_variables_freq )
      CALL MATLAB_fill_component( pointer, 'sparsify_rows_freq',               &
                                  PRESOLVE_control%sparsify_rows_freq )
      CALL MATLAB_fill_component( pointer, 'max_fill',                         &
                                  PRESOLVE_control%max_fill )
      CALL MATLAB_fill_component( pointer, 'transf_file_nbr',                  &
                                  PRESOLVE_control%transf_file_nbr )
      CALL MATLAB_fill_component( pointer, 'transf_buffer_size',               &
                                  PRESOLVE_control%transf_buffer_size )
      CALL MATLAB_fill_component( pointer, 'transf_file_status',               &
                                  PRESOLVE_control%transf_file_status )
      CALL MATLAB_fill_component( pointer, 'y_sign',                           &
                                  PRESOLVE_control%y_sign )
      CALL MATLAB_fill_component( pointer, 'inactive_y',                       &
                                  PRESOLVE_control%inactive_y )
      CALL MATLAB_fill_component( pointer, 'z_sign',                           &
                                  PRESOLVE_control%z_sign )
      CALL MATLAB_fill_component( pointer, 'inactive_z',                       &
                                  PRESOLVE_control%inactive_z )
      CALL MATLAB_fill_component( pointer, 'final_x_bounds',                   &
                                  PRESOLVE_control%final_x_bounds )
      CALL MATLAB_fill_component( pointer, 'final_z_bounds',                   &
                                  PRESOLVE_control%final_z_bounds )
      CALL MATLAB_fill_component( pointer, 'final_c_bounds',                   &
                                  PRESOLVE_control%final_c_bounds )
      CALL MATLAB_fill_component( pointer, 'final_y_bounds',                   &
                                  PRESOLVE_control%final_y_bounds )
      CALL MATLAB_fill_component( pointer, 'check_primal_feasibility',         &
                                  PRESOLVE_control%check_primal_feasibility )
      CALL MATLAB_fill_component( pointer, 'check_dual_feasibility',           &
                                  PRESOLVE_control%check_dual_feasibility )
      CALL MATLAB_fill_component( pointer, 'c_accuracy',                       &
                                  PRESOLVE_control%c_accuracy )
      CALL MATLAB_fill_component( pointer, 'z_accuracy',                       &
                                  PRESOLVE_control%z_accuracy )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  PRESOLVE_control%infinity )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  PRESOLVE_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'min_rel_improve',                  &
                                  PRESOLVE_control%min_rel_improve )
      CALL MATLAB_fill_component( pointer, 'max_growth_factor',                &
                                  PRESOLVE_control%max_growth_factor )
      CALL MATLAB_fill_component( pointer, 'dual_transformations',             &
                                  PRESOLVE_control%dual_transformations )
      CALL MATLAB_fill_component( pointer, 'redundant_xc',                     &
                                  PRESOLVE_control%redundant_xc )
      CALL MATLAB_fill_component( pointer, 'get_q',                            &
                                  PRESOLVE_control%get_q )
      CALL MATLAB_fill_component( pointer, 'get_f',                            &
                                  PRESOLVE_control%get_f )
      CALL MATLAB_fill_component( pointer, 'get_g',                            &
                                  PRESOLVE_control%get_g )
      CALL MATLAB_fill_component( pointer, 'get_H',                            &
                                  PRESOLVE_control%get_H )
      CALL MATLAB_fill_component( pointer, 'get_A',                            &
                                  PRESOLVE_control%get_A )
      CALL MATLAB_fill_component( pointer, 'get_x',                            &
                                  PRESOLVE_control%get_x )
      CALL MATLAB_fill_component( pointer, 'get_x_bounds',                     &
                                  PRESOLVE_control%get_x_bounds )
      CALL MATLAB_fill_component( pointer, 'get_z',                            &
                                  PRESOLVE_control%get_z )
      CALL MATLAB_fill_component( pointer, 'get_z_bounds',                     &
                                  PRESOLVE_control%get_z_bounds )
      CALL MATLAB_fill_component( pointer, 'get_c',                            &
                                  PRESOLVE_control%get_c )
      CALL MATLAB_fill_component( pointer, 'get_c_bounds',                     &
                                  PRESOLVE_control%get_c_bounds )
      CALL MATLAB_fill_component( pointer, 'get_y',                            &
                                  PRESOLVE_control%get_y )
      CALL MATLAB_fill_component( pointer, 'get_y_bounds',                     &
                                  PRESOLVE_control%get_y_bounds )
      CALL MATLAB_fill_component( pointer, 'transf_file_name',                 &
                                  PRESOLVE_control%transf_file_name )


      RETURN

!  End of subroutine PRESOLVE_matlab_control_get

      END SUBROUTINE PRESOLVE_matlab_control_get

! P R E S O L V E _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E

      SUBROUTINE PRESOLVE_matlab_inform_create( struct, PRESOLVE_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold PRESOLVE_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  PRESOLVE_pointer - PRESOLVE pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( PRESOLVE_pointer_type ) :: PRESOLVE_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 2
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'nbr_transforms          ' /)
!          'status                  ', 'nbr_transforms          ',             &
!          'message                 ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name,                         &
                                         PRESOLVE_pointer%pointer,             &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        PRESOLVE_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( PRESOLVE_pointer%pointer,          &
        'status', PRESOLVE_pointer%status )
      CALL MATLAB_create_integer_component( PRESOLVE_pointer%pointer,          &
         'nbr_transforms', PRESOLVE_pointer%nbr_transforms )
!     CALL MATLAB_create_char_component( PRESOLVE_pointer%pointer,             &
!       'message', 3, PRESOLVE_pointer%message )

      RETURN

!  End of subroutine PRESOLVE_matlab_inform_create

      END SUBROUTINE PRESOLVE_matlab_inform_create

!-  P R E S O L V E _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E  -

      SUBROUTINE PRESOLVE_matlab_inform_get( PRESOLVE_inform, PRESOLVE_pointer )

!  --------------------------------------------------------------

!  Set PRESOLVE_inform values from matlab pointers

!  Arguments

!  PRESOLVE_inform - PRESOLVE inform structure
!  PRESOLVE_pointer - PRESOLVE pointer structure

!  --------------------------------------------------------------

      TYPE ( PRESOLVE_inform_type ) :: PRESOLVE_inform
      TYPE ( PRESOLVE_pointer_type ) :: PRESOLVE_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( PRESOLVE_inform%status,                         &
                               mxGetPr( PRESOLVE_pointer%status ) )
      CALL MATLAB_copy_to_ptr( PRESOLVE_inform%nbr_transforms,                 &
                               mxGetPr( PRESOLVE_pointer%nbr_transforms ) )
!     CALL MATLAB_copy_to_ptr( PRESOLVE_pointer%pointer,                       &
!                              'message', PRESOLVE_inform%message( 1 : 3 ) )

      RETURN

!  End of subroutine PRESOLVE_matlab_inform_get

      END SUBROUTINE PRESOLVE_matlab_inform_get

!-*- E N D  o f  G A L A H A D _ P R E S O L V E _ T Y P E S   M O D U L E  -*-

    END MODULE GALAHAD_PRESOLVE_MATLAB_TYPES




